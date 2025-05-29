import logging
import math
import multiprocessing
import time

import cv2
import ffmpeg
import numpy as np
import requests
from requests.exceptions import RequestException

from tuatini.utils.exceptions import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError


class OpenCVCamera:
    def __init__(self, device, capture_fps, capture_width, capture_height, rotation):
        self.device = device
        self.capture_fps = capture_fps
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.rotation = rotation

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
    def __init__(self, output_device, ip, port, capture_fps, capture_width, capture_height, rotation):
        self.ip = ip
        self.port = port
        self.output_device = output_device
        self.capture_fps = capture_fps
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.rotation = rotation
        self.stream_process = None
        self.camera = None

    def connect(self):
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

            logging.info(f"Connecting to IP camera from {self.ip}:{self.port} to {self.output_device}")
            # Create and start the process
            self.stream_process = multiprocessing.Process(
                target=self._run_stream,
                daemon=True,  # Process will be terminated when main program exits
            )
            self.stream_process.start()

            # Give the process a moment to start and check if it's still alive
            time.sleep(0.5)
            if not self.stream_process.is_alive():
                raise RuntimeError("Failed to start video stream process")
            logging.info(f"Video stream accessible at {self.output_device} on the system")

            # Now that a stream is running on the output device, we can connect OpenCV to it
            self.camera = OpenCVCamera(
                self.output_device, self.capture_fps, self.capture_width, self.capture_height, self.rotation
            )
            self.camera.connect()
        except KeyboardInterrupt:
            logging.info("Shutting down IP camera stream...")
        except Exception as e:
            raise e
        finally:
            # Clean up the process
            if self.stream_process and self.stream_process.is_alive():
                self.stream_process.terminate()
                self.stream_process.join()

    def disconnect(self):
        if self.stream_process and self.stream_process.is_alive():
            self.stream_process.terminate()
            self.stream_process.join()

    def read(self):
        return self.camera.read()

    def _run_stream(self):
        """Run the FFmpeg stream in a separate process.

        Args:
            vcam_ip (str): IP address of the video stream
            vcam_local_path (str): Local path where the video stream will be accessible
        """
        try:
            stream = (
                ffmpeg.input(f"http://{self.ip}:{self.port}/video")
                .filter("scale", self.capture_width, self.capture_height)
                .filter("fps", fps=self.capture_fps)
                .filter("format", "yuyv422")
                .output(self.output_device, f="v4l2")
                .overwrite_output()
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )
            stream.wait()  # Wait for the stream to complete
        except ffmpeg.Error as e:
            logging.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            raise
