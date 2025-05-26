from pathlib import Path

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

    def teleop_step(self, record_data=False):
        pass
