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


def start_smartphone_stream(vcam_ip, vcam_port, vcam_local_path):
    """Start the smartphone video stream in a separate process.

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
