import logging
import math
import multiprocessing
import os
import subprocess
import time

import cv2
import ffmpeg
import numpy as np
import requests
from requests.exceptions import RequestException

from tuatini.utils.exceptions import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from tuatini.utils.time import capture_timestamp_utc


class OpenCVCamera:
    def __init__(self, device, capture_fps, capture_width, capture_height, rotation, color_mode="rgb"):
        self.device = device
        self.capture_fps = capture_fps
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.rotation = rotation
        self.color_mode = color_mode

        self.camera = None
        self.is_connected = False

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(f"OpenCVCamera({self.device}) is already connected.")

        camera_idx = self.device

        # Use 1 thread to avoid blocking the main thread. Especially useful during data collection
        # when other threads are used to save the images.
        cv2.setNumThreads(1)

        if self.rotation == -90:
            self.rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif self.rotation == 90:
            self.rotation = cv2.ROTATE_90_CLOCKWISE
        elif self.rotation == 180:
            self.rotation = cv2.ROTATE_180

        backend = cv2.CAP_V4L2  # Used on Linux

        # First create a temporary camera trying to access `camera_index`,
        # and verify it is a valid camera by calling `isOpened`.
        tmp_camera = cv2.VideoCapture(camera_idx, backend)
        is_camera_open = tmp_camera.isOpened()
        # Release camera to make it accessible for `find_camera_indices`
        tmp_camera.release()
        del tmp_camera

        if not is_camera_open:
            raise OSError(f"Can't access OpenCVCamera({camera_idx}).")

        self.camera = cv2.VideoCapture(camera_idx, backend)
        if not self.camera.isOpened():  # Add a final check
            raise OSError(f"Failed to open OpenCVCamera({camera_idx}) for use, even after initial checks passed.")

        if self.capture_fps is not None:
            self.camera.set(cv2.CAP_PROP_FPS, self.capture_fps)
        if self.capture_width is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        if self.capture_height is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)

        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Using `math.isclose` since actual fps can be a float (e.g. 29.9 instead of 30)
        if self.capture_fps is not None and not math.isclose(self.capture_fps, actual_fps, rel_tol=1e-3):
            # Using `OSError` since it's a broad that encompasses issues related to device communication
            raise OSError(
                f"Can't set fps={self.capture_fps} for OpenCVCamera({camera_idx}). Actual value is {actual_fps}."
            )
        if self.capture_width is not None and not math.isclose(self.capture_width, actual_width, rel_tol=1e-3):
            raise OSError(
                f"Can't set capture_width={self.capture_width} for OpenCVCamera({camera_idx}). Actual value is {actual_width}."
            )
        if self.capture_height is not None and not math.isclose(self.capture_height, actual_height, rel_tol=1e-3):
            raise OSError(
                f"Can't set capture_height={self.capture_height} for OpenCVCamera({camera_idx}). Actual value is {actual_height}."
            )

        self.capture_fps = round(actual_fps)
        self.capture_width = round(actual_width)
        self.capture_height = round(actual_height)

        self.is_connected = True

    def read(self, temporary_color_mode: str | None = None) -> np.ndarray:
        """Read a frame from the camera returned in the format (height, width, channels)
        (e.g. 480 x 640 x 3), contrarily to the pytorch format which is channel first.

        Note: Reading a frame is done every `camera.fps` times per second, and it is blocking.
        If you are reading data from other sensors, we advise to use `camera.async_read()` which is non blocking version of `camera.read()`.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )

        start_time = time.perf_counter()

        ret, color_image = self.camera.read()

        if not ret:
            raise OSError(f"Can't capture color image from camera {self.camera_index}.")

        requested_color_mode = self.color_mode if temporary_color_mode is None else temporary_color_mode

        if requested_color_mode not in ["rgb", "bgr"]:
            raise ValueError(f"Expected color values are 'rgb' or 'bgr', but {requested_color_mode} is provided.")

        # OpenCV uses BGR format as default (blue, green, red) for all operations, including displaying images.
        # However, Deep Learning framework such as LeRobot uses RGB format as default to train neural networks,
        # so we convert the image color from BGR to RGB.
        if requested_color_mode == "rgb":
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        h, w, _ = color_image.shape
        # Check against capture_height/width before rotation
        if h != self.capture_height or w != self.capture_width:
            raise OSError(
                f"Can't capture color image with expected height and width ({self.height} x {self.width}). ({h} x {w}) returned instead."
            )

        if self.rotation is not None:
            color_image = cv2.rotate(color_image, self.rotation)

        # log the number of seconds it took to read the image
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

        # log the utc time at which the image was received
        self.logs["timestamp_utc"] = capture_timestamp_utc()

        self.color_image = color_image

        return color_image


class IPCamera:
    def __init__(self, ip, port, capture_fps, capture_width, capture_height, rotation):
        self.ip = ip
        self.port = port
        self.capture_fps = capture_fps
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.rotation = rotation
        self.stream_process = None
        self.camera = None
        self.is_connected = False

        self._error_queue = multiprocessing.Queue()

    @staticmethod
    def find_available_v4l2loopback_device(start_index=0, max_devices_to_check=16):
        """
        Finds an available v4l2loopback device.

        Args:
            start_index (int): The video index to start checking from (e.g., 0 for /dev/video0).
            max_devices_to_check (int): How many /dev/videoX devices to check.

        Returns:
            str: The path to an available v4l2loopback device (e.g., "/dev/video10"),
                or None if no suitable device is found.
        """
        logging.info("Searching for an available v4l2loopback device...")
        # Prefer devices explicitly created by v4l2loopback with higher numbers
        # but check all for robustness.
        # We can iterate through existing /dev/video* devices.

        for i in range(start_index, start_index + max_devices_to_check):
            device_path = f"/dev/video{i}"
            if not os.path.exists(device_path):
                continue

            try:
                # Check if it's a v4l2loopback device
                process = subprocess.run(
                    ["v4l2-ctl", "-D", "-d", device_path], capture_output=True, text=True, check=True, timeout=2
                )
                output = process.stdout.lower()
                # Look for "v4l2 loopback" in driver or card type
                if "v4l2 loopback" in output:
                    logging.info(f"Found v4l2loopback device: {device_path}")
                    # Basic check: Does it already list formats indicating it's actively being written to?
                    # This is a heuristic. A device with exclusive_caps=1 will typically not list
                    # many formats until something writes to it.
                    # If it *does* list many, it might be in use.
                    # For now, we'll just return the first v4l2loopback device found.
                    # A more robust check would be to try and use it and handle "device busy".
                    return device_path
                else:
                    logging.debug(f"{device_path} is not a v4l2loopback device. Output: {output.strip()}")

            except subprocess.CalledProcessError as e:
                logging.warning(f"Error checking {device_path}: {e.stderr}")
            except subprocess.TimeoutExpired:
                logging.warning(f"Timeout checking {device_path}")
            except FileNotFoundError:
                logging.error("v4l2-ctl command not found. Please ensure it is installed.")
                return None  # Cannot check without v4l2-ctl

        logging.warning("No available v4l2loopback device found.")
        return None

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(f"IPCamera({self.ip}:{self.port}) is already connected.")

        # Start the video stream in a separate process
        try:
            # Check if IP camera is accessible
            camera_url = f"http://{self.ip}:{self.port}/"
            try:
                with requests.get(camera_url, timeout=2) as response:
                    if response.status_code != 200:
                        raise RuntimeError(f"IP camera returned status code {response.status_code}")
            except RequestException as e:
                raise RuntimeError(f"Failed to connect to IP camera at {camera_url}: {str(e)}")

            output_device = IPCamera.find_available_v4l2loopback_device()
            logging.info(f"Connecting to IP camera from {self.ip}:{self.port} to {output_device}")
            # Create and start the process
            self.stream_process = multiprocessing.Process(
                target=self._run_stream,
                args=(self._error_queue, output_device),  # Pass the queue to the target method
                daemon=True,
            )
            self.stream_process.start()

            # Give the process a moment to start and check if it's still alive
            time.sleep(0.5)
            if not self._error_queue.empty():
                err_from_subprocess = self._error_queue.get_nowait()
                # err_from_subprocess is now the string we put, or the exception object
                raise RuntimeError(f"FFmpeg subprocess failed: {err_from_subprocess}") from (
                    err_from_subprocess if isinstance(err_from_subprocess, Exception) else None
                )

            if not self.stream_process.is_alive():
                raise RuntimeError("Failed to start video stream process")

            # Now that a stream is running on the output device, we can connect OpenCV to it
            self.camera = OpenCVCamera(
                output_device, self.capture_fps, self.capture_width, self.capture_height, self.rotation
            )
            self.camera.connect()
            self.is_connected = True
            logging.info(f"Video stream accessible at {self.output_device} on the system")
        except KeyboardInterrupt:
            logging.info("Shutting down IP camera stream...")
            raise
        except Exception as e:
            logging.error(f"Error during IPCamera connect: {e}")
            # Clean up the process if connection fails
            if self.stream_process and self.stream_process.is_alive():
                logging.info("Terminating ffmpeg stream process due to error.")
                self.stream_process.terminate()
                self.stream_process.join(timeout=1.0)  # Add timeout to join
            elif self.stream_process:  # Process already exited
                self.stream_process.join(timeout=1.0)  # Ensure it's cleaned up
            self.stream_process = None  # Clear the process attribute
            raise

    def disconnect(self):
        if not self.is_connected:
            return

        if self.camera is not None:
            self.camera.disconnect()
            self.camera = None

        if self.stream_process and self.stream_process.is_alive():
            self.stream_process.terminate()
            self.stream_process.join()
            self.stream_process = None

        self.is_connected = False

    def read(self, temporary_color_mode: str | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"IPCamera({self.ip}:{self.port}) is not connected. Try running `camera.connect()` first."
            )
        return self.camera.read(temporary_color_mode)

    def _run_stream(self, error_queue: multiprocessing.Queue, actual_output_device: str):  # Pass queue to target
        """Run the FFmpeg stream in a separate process."""
        try:
            stream = (
                ffmpeg.input(f"http://{self.ip}:{self.port}/video")
                .filter("scale", self.capture_width, self.capture_height)
                .filter("fps", fps=self.capture_fps)
                .filter("format", "yuyv422")  # This might be the problematic part for v4l2loopback
                .output(actual_output_device, f="v4l2")
                .overwrite_output()
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )
            # This wait() will block until ffmpeg finishes or errors.
            # If it errors, ffmpeg.Error will be raised.
            _, stderr = stream.wait()  # Capture stdout and stderr

            # If ffmpeg completes but had non-fatal errors in stderr,
            # or if it exited with a non-zero code not caught by ffmpeg.Error
            if stream.process.returncode != 0:
                # Construct an error message or a custom exception
                err_msg = f"FFmpeg process exited with code {stream.process.returncode}."
                if stderr:
                    err_msg += f" Stderr: {stderr.decode(errors='ignore')}"
                error_queue.put(RuntimeError(err_msg))  # Put a generic error if not ffmpeg.Error
                return  # Terminate process

        except ffmpeg.Error as e:
            # This is where ffmpeg-python catches explicit ffmpeg errors
            errMsg = f"FFmpeg error in subprocess: {e.stderr.decode(errors='ignore') if e.stderr else str(e)}"
            logging.error(errMsg)
            error_queue.put(RuntimeError(errMsg))  # Put the detailed error message or the exception itself
            # No 'raise' here, allow the process to terminate naturally after putting error in queue
        except Exception as e:
            # Catch any other unexpected errors in the subprocess
            errMsg = f"Unexpected error in _run_stream: {str(e)}"
            logging.error(errMsg)
            error_queue.put(RuntimeError(errMsg))
