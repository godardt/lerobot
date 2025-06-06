import time
from pathlib import Path

import click
import yaml

from tuatini.devices.so_100 import SO100Robot
from tuatini.utils.logs import init_logging, init_rerun, log_control_info, log_rr_event
from tuatini.utils.time import busy_wait

root_dir = Path(__file__).parent.parent


def _teleoperate(robot: SO100Robot, control_time_s=float("inf"), fps=30):
    if not robot.is_connected:
        robot.connect()

    timestamp = 0
    start_episode_t = time.perf_counter()

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        observation, action = robot.teleop_step()

        log_rr_event(action, observation)
        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

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

    # Since it's only teleop we can run at higher FPS
    _teleoperate(robot, fps=60)
    print("Shutting down...")


if __name__ == "__main__":
    main()
