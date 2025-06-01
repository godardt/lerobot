import logging
import os
import time
from datetime import datetime
from pathlib import Path

import click
import rerun as rr
import yaml

from tuatini.datasets.lerobot import LeRobotDatasetMetadata
from tuatini.devices.so_100 import SO100Robot
from tuatini.utils.control import log_control_info

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


def _init_rerun(viewer_ip, viewer_port, session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop.

    Args:
        control_config: Configuration determining data display and robot type.
        session_name: Rerun session name. Defaults to "lerobot_control_loop".

    Raises:
        ValueError: If viewer IP is missing for non-remote configurations with display enabled.
    """
    if viewer_ip and viewer_port:
        # Configure Rerun flush batch size default to 8KB if not set
        batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
        os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size

        # Initialize Rerun based on configuration
        rr.init(session_name)
        logging.info(f"Connecting to viewer at {viewer_ip}:{viewer_port}")
        rr.connect_tcp(f"{viewer_ip}:{viewer_port}")


# TODO: teleoperate and record_dataset are comingled in this function,
# record_dataset should be a callback to teleoperate
def _teleoperate(
    robot: SO100Robot,
    record_data=False,
    control_time_s=float("inf"),
    dataset: LeRobotDatasetMetadata = None,
    single_task: None | str = None,
):
    if not robot.is_connected:
        robot.connect()

    timestamp = 0
    start_episode_t = time.perf_counter()

    # TODO: Those fps originally come from the control argument, not the cameras.
    # Probably represent the fps rate of the recording?
    fps = 30

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        observation, action = robot.teleop_step(record_data=record_data)
        if dataset is not None:
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(dt_s, fps=fps)

        timestamp = time.perf_counter() - start_episode_t


@click.command("Straightforward way to teleoperate the SO-100 robot")
@click.option(
    "--config", type=str, help="Config file for the robot", default=str(root_dir / "config" / "SO-100_ROG.yaml")
)
def main(config):
    init_logging()
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    robot = SO100Robot(config["robot"])
    robot.connect()

    rerun_config = config.get("rerun")
    # TODO: Add rerun config
    # _init_rerun(rerun_config.get("viewer_ip"), rerun_config.get("viewer_port"))

    _teleoperate(robot, record_data=True)
    print("Shutting down...")


if __name__ == "__main__":
    main()
