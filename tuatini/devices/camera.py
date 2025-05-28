import logging
import multiprocessing
import time

import ffmpeg
import requests
from requests.exceptions import RequestException


class OpenCVCamera:
    def __init__(self, device, fps, width, height):
        self.device = device
        self.fps = fps
        self.width = width
        self.height = height

    def connect(self):
        pass


class IPCamera:
    def __init__(self, output_device, ip, port, fps, width, height):
        self.ip = ip
        self.port = port
        self.output_device = output_device
        self.fps = fps
        self.width = width
        self.height = height
        self.stream_process = None

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
        except KeyboardInterrupt:
            logging.info("Shutting down IP camera stream...")
        except Exception as e:
            raise e
        finally:
            # Clean up the process
            if self.stream_process and self.stream_process.is_alive():
                self.stream_process.terminate()
                self.stream_process.join()
        return self.stream_process

    def disconnect(self):
        if self.stream_process and self.stream_process.is_alive():
            self.stream_process.terminate()
            self.stream_process.join()

    def _run_stream(self):
        """Run the FFmpeg stream in a separate process.

        Args:
            vcam_ip (str): IP address of the video stream
            vcam_local_path (str): Local path where the video stream will be accessible
        """
        try:
            stream = (
                ffmpeg.input(f"http://{self.ip}:{self.port}/video")
                .filter("scale", self.width, self.height)
                .filter("fps", fps=self.fps)
                .filter("format", "yuyv422")
                .output(self.output_device, f="v4l2")
                .overwrite_output()
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )
            stream.wait()  # Wait for the stream to complete
        except ffmpeg.Error as e:
            logging.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            raise
