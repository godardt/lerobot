import time
from pathlib import Path

import click
import yaml

from tuatini.datasets.lerobot import LeRobotDatasetMetadata
from tuatini.devices.so_100 import SO100Robot
from tuatini.utils.logs import init_logging, init_rerun, log_control_info, log_rr_event

root_dir = Path(__file__).parent.parent


def _record_dataset(
    robot: SO100Robot,
    dataset_recording_config: dict,
    record_data=False,
    control_time_s=float("inf"),
    dataset: LeRobotDatasetMetadata = None,
    single_task: None | str = None,
):
    if not robot.is_connected:
        robot.connect()

    timestamp = 0
    start_episode_t = time.perf_counter()
    fps = dataset_recording_config.get("fps", 30)

    for i in reversed(range(3)):
        print(f"Teleoperating robot in {i + 1} seconds")
        time.sleep(1)

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        observation, action = robot.teleop_step(record_data=record_data)
        if dataset is not None:
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)

        log_rr_event(action, observation)
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

    init_rerun()

    _record_dataset(robot, config.get("dataset_recording"), record_data=True)
    print("Shutting down...")


if __name__ == "__main__":
    main()
