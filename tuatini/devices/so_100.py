import time
from pathlib import Path

import torch

from tuatini.motors.feetch import FeetechMotorsBus


class SO100Robot:
    def __init__(self, config):
        self.config = config
        # name: (index, model)
        self.motors_bus_configs = {
            "shoulder_pan": (1, "sts3215"),
            "shoulder_lift": (2, "sts3215"),
            "elbow_flex": (3, "sts3215"),
            "wrist_flex": (4, "sts3215"),
            "wrist_roll": (5, "sts3215"),
            "gripper": (6, "sts3215"),
        }

        # TODO is leader_arms"]["main"] correct or should I iterate?
        leader_arms = self.make_motors_buses_from_configs(self.config["leader_arms"])
        follower_arms = self.make_motors_buses_from_configs(self.config["follower_arms"])
        cameras = self.make_cameras_from_configs(self.config["cameras"])

        self.calibration_dir = Path(self.config["calibration_dir"])

    def make_motors_buses_from_configs(self, arms_config):
        motors_buses = {}
        for arm_name, data in arms_config.items():
            motors_buses[arm_name] = FeetechMotorsBus(self.motors_bus_configs, data["device"])

        return motors_buses

    def make_cameras_from_configs(self, config):
        pass

    @property
    def is_connected(self):
        return (
            all(arm.is_connected for arm in self.leader_arms.values())
            and all(arm.is_connected for arm in self.follower_arms.values())
            and all(camera.is_connected for camera in self.cameras)
        )

    def connect(self):
        for arm_name, arm in self.leader_arms.items():
            arm.connect()
        for arm_name, arm in self.follower_arms.items():
            arm.connect()
        for camera in self.cameras:
            camera.connect()

    def teleop_step(self, record_data=False) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].read("Present_Position")
            leader_pos[name] = torch.from_numpy(leader_pos[name])
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t
