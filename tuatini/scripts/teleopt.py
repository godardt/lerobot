import logging
import multiprocessing
import time
from datetime import datetime
from pathlib import Path

import click
import ffmpeg
import rerun as rr
import yaml

root_dir = Path(__file__).parent.parent


def init_logging():
    def custom_format(record):
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fnameline = f"{record.pathname}:{record.lineno}"
        message = f"{record.levelname} {dt} {fnameline[-15:]:>15} {record.msg}"
        return message

    logging.basicConfig(level=logging.INFO)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    formatter = logging.Formatter()
    formatter.format = custom_format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)


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


def _start_smartphone_stream(vcam_ip, vcam_port, vcam_local_path):
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


def _get_smartphone_config(config_path):
    """Get viewer IP and port from config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    smartphone_config = config["smartphone_camera"]
    ip = smartphone_config["vcam_ip"]
    port = smartphone_config["vcam_port"]
    output_path = smartphone_config["vcam_output_path"]
    return ip, port, output_path


@click.command("Straightforward way to teleoperate the SO-100 robot")
@click.option("--config", type=str, help="Config file for the robot", default=str(root_dir / "config" / "SO-100.yaml"))
def main(config):
    init_logging()
    vcam_ip, vcam_port, vcam_output_path = _get_smartphone_config(config)
    logging.info(f"Connecting to smartphone camera from {vcam_ip}:{vcam_port} to {vcam_output_path}")

    # Start the video stream in a separate process
    stream_process = _start_smartphone_stream(vcam_ip, vcam_port, vcam_output_path)

    # TODO check connection to the wrist cam
    

    try:
        # Your main program logic here
        while True:  # Keep the main thread alive
            print("main loop")
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        # Clean up the process
        if stream_process and stream_process.is_alive():
            stream_process.terminate()
            stream_process.join()


if __name__ == "__main__":
    main()
