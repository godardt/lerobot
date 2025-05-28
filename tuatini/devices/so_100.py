import time
from pathlib import Path

import torch

from tuatini.devices.camera import IPCamera, OpenCVCamera
from tuatini.motors.feetch import FeetechMotorsBus


class RobotDeviceNotConnectedError(Exception):
    """Exception raised when the robot device is not connected."""

    def __init__(self, message="This robot device is not connected. Try calling `robot_device.connect()` first."):
        self.message = message
        super().__init__(self.message)


class RobotDeviceAlreadyConnectedError(Exception):
    """Exception raised when the robot device is already connected."""

    def __init__(
        self,
        message="This robot device is already connected. Try not calling `robot_device.connect()` twice.",
    ):
        self.message = message
        super().__init__(self.message)


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

        # TODO is leader_arms"]["main"] correct or should I iterate because there can be many?
        self.leader_arms = self.make_motors_buses_from_configs(self.config["leader_arms"])
        self.follower_arms = self.make_motors_buses_from_configs(self.config["follower_arms"])
        self.cameras = self.make_cameras_from_configs(self.config["cameras"])

        self.calibration_dir = Path(self.config["calibration_dir"])
        self.is_connected = False

    @property
    def num_cameras(self):
        return len(self.cameras)

    def make_motors_buses_from_configs(self, arms_config):
        motors_buses = {}
        for arm_name, data in arms_config.items():
            motors_buses[arm_name] = FeetechMotorsBus(self.motors_bus_configs, data["device"])

        return motors_buses

    def make_cameras_from_configs(self, config):
        cameras = {}
        for cam_name, cam_config in config.items():
            fps = cam_config["fps"]
            width = cam_config["width"]
            height = cam_config["height"]
            device = cam_config["device"]

            if cam_config["type"] == "ip_camera":
                ip = cam_config["vcam_ip"]
                port = cam_config["vcam_port"]
                cameras[cam_name] = IPCamera(device, ip, port, fps, width, height)
            elif cam_config["type"] == "opencv":
                cameras[cam_name] = OpenCVCamera(device, fps, width, height)

        return cameras

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        if not self.leader_arms and not self.follower_arms and not self.cameras:
            raise ValueError(
                "ManipulatorRobot doesn't have any device to connect. See example of usage in docstring of the class."
            )

        # Connect the arms
        for name in self.follower_arms:
            print(f"Connecting {name} follower arm.")
            self.follower_arms[name].connect()
        for name in self.leader_arms:
            print(f"Connecting {name} leader arm.")
            self.leader_arms[name].connect()

        if self.robot_type in ["koch", "koch_bimanual", "aloha"]:
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
        elif self.robot_type in ["so100", "so101", "moss", "lekiwi"]:
            from lerobot.common.robot_devices.motors.feetech import TorqueMode

        # We assume that at connection time, arms are in a rest position, and torque can
        # be safely disabled to run calibration and/or set robot preset configurations.
        for name in self.follower_arms:
            self.follower_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
        for name in self.leader_arms:
            self.leader_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)

        self.activate_calibration()

        # Set robot preset (e.g. torque in leader gripper for Koch v1.1)
        if self.robot_type in ["koch", "koch_bimanual"]:
            self.set_koch_robot_preset()
        elif self.robot_type == "aloha":
            self.set_aloha_robot_preset()
        elif self.robot_type in ["so100", "so101", "moss", "lekiwi"]:
            self.set_so100_robot_preset()

        # Enable torque on all motors of the follower arms
        for name in self.follower_arms:
            print(f"Activating torque on {name} follower arm.")
            self.follower_arms[name].write("Torque_Enable", 1)

        if self.config.gripper_open_degree is not None:
            if self.robot_type not in ["koch", "koch_bimanual"]:
                raise NotImplementedError(
                    f"{self.robot_type} does not support position AND current control in the handle, which is require to set the gripper open."
                )
            # Set the leader arm in torque mode with the gripper motor set to an angle. This makes it possible
            # to squeeze the gripper and have it spring back to an open position on its own.
            for name in self.leader_arms:
                self.leader_arms[name].write("Torque_Enable", 1, "gripper")
                self.leader_arms[name].write("Goal_Position", self.config.gripper_open_degree, "gripper")

        # Check both arms can be read
        for name in self.follower_arms:
            self.follower_arms[name].read("Present_Position")
        for name in self.leader_arms:
            self.leader_arms[name].read("Present_Position")

        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()

        self.is_connected = True

    def teleop_step(self, record_data=False) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("ManipulatorRobot is not connected. You need to run `robot.connect()`.")
        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].read("Present_Position")
            leader_pos[name] = torch.from_numpy(leader_pos[name])
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t
