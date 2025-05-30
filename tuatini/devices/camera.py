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
        self.device_path = device  # Store original path for clarity
        # OpenCV expects an integer index or a string device path.
        # If it's like "/dev/video0", use it directly. If it's just "0", convert to int.
        try:
            self.camera_index = int(device)
        except ValueError:
            self.camera_index = device  # Use as string path, e.g., /dev/video0

        self.capture_fps = capture_fps
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.rotation_code = None  # OpenCV rotation code
        self.color_mode = color_mode

        self.camera: cv2.VideoCapture | None = None
        self.is_connected = False
        self.logs = {}  # Initialize logs

        if rotation == -90:
            self.rotation_code = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif rotation == 90:
            self.rotation_code = cv2.ROTATE_90_CLOCKWISE
        elif rotation == 180:
            self.rotation_code = cv2.ROTATE_180
        elif rotation not in [None, 0]:  # 0 or None means no rotation
            logging.warning(f"Invalid rotation value {rotation} for OpenCVCamera. No rotation will be applied.")

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(f"OpenCVCamera({self.device_path}) is already connected.")

        logging.info(f"Connecting to OpenCVCamera device: {self.device_path}")
        # Use 1 thread to avoid blocking the main thread.
        cv2.setNumThreads(1)

        backend = cv2.CAP_V4L2  # Explicitly use V4L2 on Linux

        # Temporary camera check (optional but good for early failure)
        # This can sometimes be problematic if the device doesn't like rapid open/close
        # If issues persist, you might remove this tmp check and rely on the main open.
        tmp_camera = None
        try:
            logging.debug(f"Attempting temporary open for {self.camera_index}")
            tmp_camera = cv2.VideoCapture(self.camera_index, backend)
            is_camera_open = tmp_camera.isOpened()
            if not is_camera_open:
                raise OSError(f"Initial check: Can't access OpenCVCamera({self.camera_index}).")
            logging.debug(f"Temporary open for {self.camera_index} successful, releasing.")
        except Exception as e:
            logging.error(f"Error during temporary camera check for {self.camera_index}: {e}")
            raise  # Re-raise the exception
        finally:
            if tmp_camera is not None:
                tmp_camera.release()
            del tmp_camera
            time.sleep(0.5)  # Give a very short time for the device to settle after release

        # Main camera connection
        self.camera = cv2.VideoCapture(self.camera_index, backend)
        if not self.camera.isOpened():
            self.camera = None  # Ensure self.camera is None if open fails
            raise OSError(
                f"Failed to open OpenCVCamera({self.camera_index}) for use, even after initial checks passed."
            )

        logging.info(f"Successfully opened OpenCVCamera({self.camera_index}). Configuring...")

        try:
            if self.capture_fps is not None:
                self.camera.set(cv2.CAP_PROP_FPS, float(self.capture_fps))
            if self.capture_width is not None:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.capture_width))
            if self.capture_height is not None:
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.capture_height))

            # Verify settings
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

            logging.info(f"Requested: FPS={self.capture_fps}, W={self.capture_width}, H={self.capture_height}")
            logging.info(f"Actual:    FPS={actual_fps}, W={actual_width}, H={actual_height}")

            # Using math.isclose since actual fps can be a float
            if self.capture_fps is not None and not math.isclose(
                self.capture_fps, actual_fps, rel_tol=0.1, abs_tol=1.0
            ):  # Relaxed tolerance for FPS
                # It's common for cameras not to hit the exact FPS, so this might be too strict
                logging.warning(
                    f"Could not set exact FPS {self.capture_fps} for OpenCVCamera({self.camera_index}). Actual: {actual_fps}."
                )
            if self.capture_width is not None and int(actual_width) != int(self.capture_width):
                raise OSError(
                    f"Can't set capture_width={self.capture_width} for OpenCVCamera({self.camera_index}). Actual: {actual_width}."
                )
            if self.capture_height is not None and int(actual_height) != int(self.capture_height):
                raise OSError(
                    f"Can't set capture_height={self.capture_height} for OpenCVCamera({self.camera_index}). Actual: {actual_height}."
                )

            # Update with actual values, rounded
            self.capture_fps = round(actual_fps) if actual_fps else self.capture_fps
            self.capture_width = round(actual_width) if actual_width else self.capture_width
            self.capture_height = round(actual_height) if actual_height else self.capture_height

            self.is_connected = True
            logging.info(f"OpenCVCamera({self.device_path}) connected successfully.")
        except Exception as e:
            logging.error(f"Error during OpenCVCamera configuration or final check: {e}")
            if self.camera is not None and self.camera.isOpened():
                self.camera.release()
            self.camera = None
            self.is_connected = False
            raise  # Re-raise the exception

    def disconnect(self):
        if not self.is_connected:
            # logging.debug(f"OpenCVCamera({self.device_path}) already disconnected or never connected.")
            return

        logging.info(f"Disconnecting OpenCVCamera({self.device_path})...")
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
            logging.info(f"OpenCVCamera({self.device_path}) released.")
        self.camera = None
        self.is_connected = False

    def read(self, temporary_color_mode: str | None = None) -> np.ndarray:
        """
        Read a frame from the camera returned in the format (height, width, channels)
        (e.g. 480 x 640 x 3), contrarily to the pytorch format which is channel first.

        Note: Reading a frame is done every `camera.fps` times per second, and it is blocking.
        If you are reading data from other sensors, we advise to use `camera.async_read()` which is non blocking version of `camera.read()`.
        """
        if not self.is_connected or self.camera is None:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.device_path}) is not connected. Try running `camera.connect()` first."
            )

        start_time = time.perf_counter()

        ret, color_image = self.camera.read()

        if not ret or color_image is None:
            # Attempt a few retries before failing hard, cameras can glitch
            for attempt in range(3):
                time.sleep(0.5)  # Small delay
                logging.warning(f"Retrying frame read for {self.device_path}, attempt {attempt + 1}")
                ret, color_image = self.camera.read()
                if ret and color_image is not None:
                    break
            else:
                raise OSError(f"Can't capture color image from camera {self.device_path} after retries.")

        requested_color_mode = self.color_mode if temporary_color_mode is None else temporary_color_mode

        if requested_color_mode not in ["rgb", "bgr"]:
            raise ValueError(f"Expected color values are 'rgb' or 'bgr', but {requested_color_mode} is provided.")

        # OpenCV uses BGR format as default (blue, green, red) for all operations, including displaying images.
        # However, Deep Learning framework such as LeRobot uses RGB format as default to train neural networks,
        # so we convert the image color from BGR to RGB.
        if requested_color_mode == "rgb" and self.color_mode != "rgb":  # Assuming OpenCV default is BGR
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        elif requested_color_mode == "bgr" and self.color_mode == "rgb":  # If internal is RGB but BGR requested
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        h, w, _ = color_image.shape
        # Check against actual capture_height/width before rotation
        # Note: self.capture_height/width are updated to actual values in connect()
        if h != self.capture_height or w != self.capture_width:
            logging.warning(
                f"Captured image dimensions ({h}x{w}) differ from expected ({self.capture_height}x{self.capture_width}) for {self.device_path}."
            )

        if self.rotation_code is not None:
            color_image = cv2.rotate(color_image, self.rotation_code)

        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time
        self.logs["timestamp_utc"] = capture_timestamp_utc()
        self.color_image = color_image

        return color_image

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class IPCamera:
    def __init__(self, ip, port, capture_fps, capture_width, capture_height, rotation, color_mode="rgb"):
        self.ip = ip
        self.port = port
        self.capture_fps = capture_fps
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.rotation = rotation  # Rotation for the OpenCVCamera reading the loopback
        self.color_mode = color_mode
        self.stream_process: multiprocessing.Process | None = None
        self.camera: OpenCVCamera | None = None  # Will be an instance of OpenCVCamera
        self.is_connected = False
        self.output_device_path: str | None = None  # Store the loopback device path used

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
        for i in range(start_index, start_index + max_devices_to_check):
            device_path = f"/dev/video{i}"
            if not os.path.exists(device_path):
                continue
            try:
                process = subprocess.run(
                    ["v4l2-ctl", "-D", "-d", device_path], capture_output=True, text=True, check=True, timeout=2
                )
                output = process.stdout.lower()
                if "v4l2 loopback" in output:
                    logging.info(f"Found v4l2loopback device: {device_path}")
                    return device_path
                else:
                    logging.debug(f"{device_path} is not a v4l2loopback device. Output: {output.strip()}")
            except subprocess.CalledProcessError as e:
                logging.warning(f"Error checking {device_path} with v4l2-ctl: {e.stderr.strip() if e.stderr else e}")
            except subprocess.TimeoutExpired:
                logging.warning(f"Timeout checking {device_path} with v4l2-ctl")
            except FileNotFoundError:
                logging.error("v4l2-ctl command not found. Please ensure it is installed.")
                return None
        logging.warning("No available v4l2loopback device found.")
        return None

    @staticmethod
    def free_v4l2loopback_device(device_path: str) -> bool:
        """
        FIXME: This is a hack to free the device using the root user. It is not a good idea to use it in production.
        Attempts to free a v4l2loopback device by completely unloading and reloading the module.
        This is the only way I found to remove it from a "stuck" state.

        Args:
            device_path (str): The path to the v4l2loopback device (e.g., "/dev/video10")

        Returns:
            bool: True if the device was successfully freed, False otherwise
        """
        if not os.path.exists(device_path):
            logging.warning(f"Device {device_path} does not exist")
            return False

        try:
            # First check if it's actually a v4l2loopback device
            process = subprocess.run(
                ["v4l2-ctl", "-D", "-d", device_path], capture_output=True, text=True, check=True, timeout=2
            )
            if "v4l2 loopback" not in process.stdout.lower():
                logging.warning(f"{device_path} is not a v4l2loopback device")
                return False

            logging.info(f"Attempting to free v4l2loopback device: {device_path}")

            # 1. Try to unload the v4l2loopback module
            try:
                subprocess.run(["sudo", "rmmod", "v4l2loopback"], check=True, timeout=5)
                logging.info("Successfully unloaded v4l2loopback module")
            except subprocess.CalledProcessError as e:
                logging.warning(f"Failed to unload v4l2loopback module: {e}")
                # Continue anyway as the module might not be loaded

            # 2. Reload the module with specific parameters
            try:
                subprocess.run(
                    [
                        "sudo",
                        "modprobe",
                        "v4l2loopback",
                        "video_nr=0",
                        'card_label="IPCamLoopbackDev0"',
                        "exclusive_caps=1",
                    ],
                    check=True,
                    timeout=5,
                )
                logging.info("Successfully reloaded v4l2loopback module with specified parameters")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to reload v4l2loopback module: {e}")
                return False

            # 3. Verify the device is now available
            try:
                # Try to open the device with OpenCV to verify it's free
                cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
                if cap.isOpened():
                    cap.release()
                    logging.info(f"Successfully freed v4l2loopback device: {device_path}")
                    return True
            except Exception as e:
                logging.warning(f"Error verifying device {device_path} is free: {e}")

            logging.warning(f"Could not fully free v4l2loopback device: {device_path}")
            return False

        except subprocess.CalledProcessError as e:
            logging.error(f"Error freeing device {device_path}: {e.stderr.strip() if e.stderr else e}")
            return False
        except subprocess.TimeoutExpired:
            logging.error(f"Timeout while trying to free device {device_path}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error while freeing device {device_path}: {e}")
            return False

    @staticmethod
    def _check_camera_worker(device_path, result_queue, timeout_s):
        """Worker for checking camera with a timeout on the read itself."""
        cap = None
        try:
            cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
            if not cap.isOpened():
                logging.info(f"Worker: Failed to open {device_path}.")
                result_queue.put(False)
                return

            logging.info(f"Worker: {device_path} opened successfully. Attempting read with timeout logic.")
            # The read itself can hang, cv2.VideoCapture doesn't offer a timeout for read
            # So, if just opening is enough, we can simplify.
            # For a more robust read check, we'd need another layer of process timeout for the read.
            # For now, if it opens, we assume a read *should* work soon after ffmpeg starts.
            # Let's try a quick read. If this hangs, this static method needs the full process timeout too.
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_s * 1000)  # May not be supported by all backends/versions
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_s * 1000)  # May not be supported

            ret, _ = cap.read()  # This read can still hang if timeouts aren't supported
            if not ret:
                logging.info(f"Worker: Opened {device_path}, but failed to read a frame initially.")
                result_queue.put(False)  # Or True if open is enough? For IPCam, read is important.
            else:
                logging.info(f"Worker: Successfully read a frame from {device_path}.")
                result_queue.put(True)
        except Exception as e:
            logging.error(f"Worker: Exception during OpenCV check for {device_path}: {e}")
            result_queue.put(False)
        finally:
            if cap is not None:
                cap.release()
            if result_queue.empty():  # Ensure something is put if an unhandled exit
                result_queue.put(False)

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(f"IPCamera({self.ip}:{self.port}) is already connected.")

        logging.info(f"Connecting to IPCamera at http://{self.ip}:{self.port}")
        # Check IP camera accessibility
        camera_url = f"http://{self.ip}:{self.port}/"  # Check base URL or a known status endpoint
        video_stream_url = f"http://{self.ip}:{self.port}/video"  # Actual stream URL for ffmpeg
        try:
            with requests.get(camera_url, timeout=3) as response:
                response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
                logging.info(f"IP camera at {camera_url} is responsive (status {response.status_code}).")
        except RequestException as e:
            raise RuntimeError(f"Failed to connect to IP camera endpoint at {camera_url}: {str(e)}")

        # Find and free a v4l2loopback device
        self.output_device_path = IPCamera.find_available_v4l2loopback_device()
        if not self.output_device_path:
            raise RuntimeError("No v4l2loopback device found or available for IPCamera.")

        # Try to free the device before using it
        if not IPCamera.free_v4l2loopback_device(self.output_device_path):
            logging.warning(
                f"Could not free v4l2loopback device {self.output_device_path}, but will attempt to use it anyway"
            )

        logging.info(f"Streaming from {video_stream_url} to {self.output_device_path}")

        # Create and start the ffmpeg stream process
        self.stream_process = multiprocessing.Process(
            target=self._run_stream,
            args=(self._error_queue, video_stream_url, self.output_device_path),
            daemon=True,  # Daemon ensures it exits if main script exits
        )
        self.stream_process.start()

        # Wait a bit for ffmpeg to start and check for immediate errors
        time.sleep(1.0)  # Give ffmpeg time to initialize or fail
        if not self._error_queue.empty():
            err_from_subprocess = self._error_queue.get_nowait()
            self._cleanup_ffmpeg_process()  # Ensure ffmpeg is stopped
            raise RuntimeError(f"FFmpeg subprocess failed on startup: {err_from_subprocess}")

        if not self.stream_process.is_alive():
            self._cleanup_ffmpeg_process()  # Ensure ffmpeg is stopped
            # Check queue again in case it put something right before exiting
            err_msg = "Unknown error."
            if not self._error_queue.empty():
                err_msg = self._error_queue.get_nowait()
            raise RuntimeError(f"FFmpeg process died unexpectedly shortly after start. Error: {err_msg}")

        # Now connect OpenCVCamera to the loopback device
        try:
            self.camera = OpenCVCamera(
                self.output_device_path,
                self.capture_fps,
                self.capture_width,
                self.capture_height,
                self.rotation,
                color_mode=self.color_mode,
            )
            self.camera.connect()
            self.is_connected = True
            logging.info(
                f"IPCamera stream from {self.ip}:{self.port} successfully connected via {self.output_device_path}"
            )
        except Exception as e:
            logging.error(f"Error connecting OpenCVCamera to loopback device {self.output_device_path}: {e}")
            self.disconnect()  # Full cleanup
            raise  # Re-raise the specific error

    def _cleanup_ffmpeg_process(self):
        if self.stream_process:
            if self.stream_process.is_alive():
                logging.info("Terminating ffmpeg stream process...")
                self.stream_process.terminate()
                self.stream_process.join(timeout=2.0)  # Wait for termination
                if self.stream_process.is_alive():
                    logging.warning("FFmpeg process did not terminate gracefully, killing.")
                    self.stream_process.kill()
                    self.stream_process.join(timeout=1.0)
            else:  # Process already exited, just join to clean up resources
                self.stream_process.join(timeout=1.0)
            self.stream_process = None

    def disconnect(self):
        if not self.is_connected and not self.stream_process:  # if neither connected nor process exists
            # logging.debug(f"IPCamera({self.ip}:{self.port}) already disconnected or never fully connected.")
            return

        logging.info(f"Disconnecting IPCamera({self.ip}:{self.port})...")
        if self.camera is not None:
            self.camera.disconnect()
            self.camera = None

        self._cleanup_ffmpeg_process()
        self.is_connected = False
        logging.info(f"IPCamera({self.ip}:{self.port}) disconnected.")

    def read(self, temporary_color_mode: str | None = None):
        if not self.is_connected or self.camera is None:
            raise RobotDeviceNotConnectedError(
                f"IPCamera({self.ip}:{self.port}) is not connected. Try running `camera.connect()` first."
            )
        # Check if ffmpeg process is still alive before reading
        if self.stream_process is None or not self.stream_process.is_alive():
            # Check error queue if ffmpeg died
            err_msg = "FFmpeg process is not running."
            if not self._error_queue.empty():
                err_msg += f" Last error: {self._error_queue.get_nowait()}"
            self.disconnect()  # Perform full cleanup
            raise RuntimeError(err_msg)

        return self.camera.read(temporary_color_mode)

    def _run_stream(self, error_queue: multiprocessing.Queue, video_url: str, loopback_device: str):
        ffmpeg_cmd_compiled_args = None
        # ffmpeg_popen_obj will now store the Popen object directly
        ffmpeg_popen_obj: subprocess.Popen | None = None
        try:
            logging.info(f"FFmpeg subprocess: Starting stream from {video_url} to {loopback_device}")

            input_opts = {}
            if video_url.startswith("rtsp://"):
                input_opts["rtsp_transport"] = "tcp"
                input_opts["stimeout"] = "10000000"
                ffmpeg_input = ffmpeg.input(video_url, **input_opts)
            elif video_url.startswith("http://") or video_url.startswith("https://"):
                input_opts["format"] = "mjpeg"
                input_opts["timeout"] = "10000000"
                ffmpeg_input = ffmpeg.input(video_url, **input_opts)
            else:
                logging.warning(f"Unknown video URL scheme for {video_url}. Proceeding with default input.")
                ffmpeg_input = ffmpeg.input(video_url)

            stream_pipeline = ffmpeg_input
            if self.capture_width is not None and self.capture_height is not None:
                stream_pipeline = stream_pipeline.filter("scale", self.capture_width, self.capture_height)
            if self.capture_fps is not None:
                stream_pipeline = stream_pipeline.filter("fps", fps=self.capture_fps)

            stream_pipeline = stream_pipeline.output(loopback_device, format="v4l2", pix_fmt="yuyv422")

            ffmpeg_cmd_instance = stream_pipeline.overwrite_output()
            ffmpeg_cmd_compiled_args = ffmpeg_cmd_instance.compile()
            logging.info(f"FFmpeg command: {' '.join(ffmpeg_cmd_compiled_args)}")
            ffmpeg_popen_obj = ffmpeg_cmd_instance.run_async(pipe_stdout=True, pipe_stderr=True)

            # This blocks until the stream stops by killing the process on the IP camera or the stream process (gracefully or not)
            stdout_bytes, stderr_bytes = ffmpeg_popen_obj.communicate()
            return_code = ffmpeg_popen_obj.returncode  # Directly from Popen object

            if return_code != 0:
                err_msg = f"FFmpeg process for {loopback_device} exited with code {return_code}."
                if stderr_bytes:
                    err_msg += f" Stderr: {stderr_bytes.decode(errors='ignore').strip()}"
                if stdout_bytes:
                    err_msg += f" Stdout: {stdout_bytes.decode(errors='ignore').strip()}"
                logging.error(err_msg)
                error_queue.put(RuntimeError(err_msg))
            else:
                logging.info(f"FFmpeg process for {loopback_device} finished gracefully (return_code 0).")
                warn_msg = f"FFmpeg stream ended for {loopback_device} (return code 0)."
                if stderr_bytes:
                    warn_msg += f" Last stderr: {stderr_bytes.decode(errors='ignore').strip()}"
                logging.warning(warn_msg)
                error_queue.put(RuntimeError(warn_msg))

        except ffmpeg.Error as e:
            errMsg = f"FFmpeg library error in subprocess setup for {loopback_device}: {str(e)}"
            if hasattr(e, "stderr") and e.stderr:
                errMsg += f" Stderr: {e.stderr.decode(errors='ignore')}"
            logging.error(errMsg)
            error_queue.put(RuntimeError(errMsg))
        except Exception as e:
            errMsg = f"Unexpected error in _run_stream for {loopback_device}: {type(e).__name__} {str(e)}"
            logging.error(errMsg, exc_info=True)  # Log full traceback
            error_queue.put(RuntimeError(errMsg))
            # If ffmpeg_popen_obj exists and is alive, try to kill it
            if ffmpeg_popen_obj and ffmpeg_popen_obj.poll() is None:  # Check Popen object directly
                logging.warning(f"Unexpected error occurred. Terminating FFmpeg process for {loopback_device}.")
                try:
                    ffmpeg_popen_obj.terminate()
                    ffmpeg_popen_obj.wait(timeout=2.0)  # Popen.wait()
                except subprocess.TimeoutExpired:
                    logging.warning(f"FFmpeg process for {loopback_device} did not terminate gracefully. Killing.")
                    ffmpeg_popen_obj.kill()
                    ffmpeg_popen_obj.wait(timeout=1.0)
                except Exception as kill_e:
                    logging.error(f"Error while trying to terminate/kill FFmpeg: {kill_e}")
        finally:
            logging.info(f"FFmpeg subprocess for {loopback_device} is terminating (reached finally block).")
            if ffmpeg_popen_obj and ffmpeg_popen_obj.returncode is None:  # Check Popen object directly
                logging.warning(
                    f"FFmpeg process for {loopback_device} seems to have been terminated externally or is still running."
                )
                if error_queue.empty():
                    error_queue.put(RuntimeError(f"FFmpeg subprocess for {loopback_device} terminated abnormally."))
            elif error_queue.empty():
                error_queue.put(
                    RuntimeError(
                        f"FFmpeg subprocess for {loopback_device} finished/failed; no specific error in queue."
                    )
                )

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
