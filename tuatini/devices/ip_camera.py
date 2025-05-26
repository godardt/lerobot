import logging
import multiprocessing
import time

import ffmpeg


def _run_stream(vcam_ip, vcam_port, vcam_local_path):
    """Run the FFmpeg stream in a separate process.

    Args:
        vcam_ip (str): IP address of the video stream
        vcam_local_path (str): Local path where the video stream will be accessible
    """
    try:
        stream = (
            ffmpeg.input(f"http://{vcam_ip}:{vcam_port}/video")
            .filter("scale", 640, 480)
            .filter("fps", fps=30)
            .filter("format", "yuyv422")
            .output(vcam_local_path, f="v4l2")
            .overwrite_output()
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        stream.wait()  # Wait for the stream to complete
    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        raise


def _run_in_process(vcam_ip, vcam_port, vcam_local_path):
    """Start the IP camera video stream in a separate process.

    Args:
        vcam_ip (str): IP address of the video stream
        vcam_local_path (str): Local path where the video stream will be accessible

    Returns:
        multiprocessing.Process: The running process

    Raises:
        RuntimeError: If the process fails to start
    """
    logging.info(f"Video stream accessible at {vcam_local_path} on the system")

    # Create and start the process
    process = multiprocessing.Process(
        target=_run_stream,
        args=(vcam_ip, vcam_port, vcam_local_path),
        daemon=True,  # Process will be terminated when main program exits
    )
    process.start()

    # Give the process a moment to start and check if it's still alive
    time.sleep(0.5)
    if not process.is_alive():
        raise RuntimeError("Failed to start video stream process")

    return process


def _get_camera_config(camera_config):
    """Get viewer IP and port from config file."""
    ip = camera_config["vcam_ip"]
    port = camera_config["vcam_port"]
    output_device = camera_config["device"]
    return ip, port, output_device


def start_stream(config):
    # Start the video stream in a separate process
    try:
        vcam_ip, vcam_port, vcam_output_device = _get_camera_config(config)
        logging.info(f"Connecting to IP camera from {vcam_ip}:{vcam_port} to {vcam_output_device}")
        stream_process = _run_in_process(vcam_ip, vcam_port, vcam_output_device)
    except KeyboardInterrupt:
        logging.info("Shutting down IP camera stream...")
    except Exception as e:
        raise e
    finally:
        # Clean up the process
        if stream_process and stream_process.is_alive():
            stream_process.terminate()
            stream_process.join()
    return stream_process
