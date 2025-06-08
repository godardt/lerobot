import logging
from pathlib import Path

import click
import yaml

from tuatini.devices.so_100 import SO100Robot
from tuatini.utils.logs import init_logging

root_dir = Path(__file__).parent.parent


@click.command("Calibrate the SO-100 robot")
@click.option(
    "--config", type=str, help="Config file for the robot", default=str(root_dir / "config" / "SO-100_ROG.yaml")
)
def main(config):
    init_logging()
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    for robot_type in ["leader_arms", "follower_arms"]:
        for arm_name, arm_config in config["robots"][robot_type].items():
            logging.info(f"Calibrating {robot_type}/{arm_name}...")
            device = arm_config["device"]
            calibration_dir = arm_config["calibration_dir"]
            robot = SO100Robot(device, calibration_dir, name=f"{robot_type}/{arm_name}")
            robot.connect(calibrated=False)
            robot.calibrate()
            robot.disconnect()

    logging.info("Calibration complete")


if __name__ == "__main__":
    main()
