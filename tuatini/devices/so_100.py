import json
import time
from pathlib import Path

import torch

from tuatini.devices.camera import IPCamera, OpenCVCamera
from tuatini.motors.feetch import FeetechMotorsBus, FeetechTorqueMode
from tuatini.utils.exceptions import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from tuatini.utils.io import get_arm_id, substitute_path_variables


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

        self.calibration_dir = Path(substitute_path_variables(self.config["calibration_dir"]))
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
            rotation = cam_config["rotation"]

            if cam_config["type"] == "ip_camera":
                ip = cam_config["vcam_ip"]
                port = cam_config["vcam_port"]
                cameras[cam_name] = IPCamera(device, ip, port, fps, width, height, rotation)
            elif cam_config["type"] == "opencv":
                cameras[cam_name] = OpenCVCamera(device, fps, width, height, rotation)

        return cameras

    def set_robot_preset(self):
        for name in self.follower_arms:
            # Mode=0 for Position Control
            self.follower_arms[name].write("Mode", 0)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.follower_arms[name].write("P_Coefficient", 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.follower_arms[name].write("I_Coefficient", 0)
            self.follower_arms[name].write("D_Coefficient", 4)
            # Close the write lock so that Maximum_Acceleration gets written to EPROM address,
            # which is mandatory for Maximum_Acceleration to take effect after rebooting.
            self.follower_arms[name].write("Lock", 0)
            # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
            # the motors. Note: this configuration is not in the official STS3215 Memory Table
            self.follower_arms[name].write("Maximum_Acceleration", 254)
            self.follower_arms[name].write("Acceleration", 254)

    def activate_calibration(self):
        """After calibration all motors function in human interpretable ranges.
        Rotations are expressed in degrees in nominal range of [-180, 180].
        """

        def load_calibration_(name, arm_type):
            # TODO: Calibration can be run here but we throw an error for matter of simplicity
            arm_id = get_arm_id(name, arm_type)
            arm_calib_path = self.calibration_dir / f"{arm_id}.json"

            if not arm_calib_path.exists():
                raise FileNotFoundError(f"Calibration file '{arm_calib_path}' not found. Please run calibration first")

            with open(arm_calib_path) as f:
                calibration = json.load(f)

            return calibration

        for name, arm in self.follower_arms.items():
            calibration = load_calibration_(name, "follower")
            arm.set_calibration(calibration)
        for name, arm in self.leader_arms.items():
            calibration = load_calibration_(name, "leader")
            arm.set_calibration(calibration)

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

        # We assume that at connection time, arms are in a rest position, and torque can
        # be safely disabled to run calibration and/or set robot preset configurations.
        for name in self.follower_arms:
            self.follower_arms[name].write("Torque_Enable", FeetechTorqueMode.DISABLED.value)
        for name in self.leader_arms:
            self.leader_arms[name].write("Torque_Enable", FeetechTorqueMode.DISABLED.value)

        self.activate_calibration()
        self.set_robot_preset()

        # Enable torque on all motors of the follower arms
        for name in self.follower_arms:
            print(f"Activating torque on {name} follower arm.")
            self.follower_arms[name].write("Torque_Enable", 1)

        # Check all arms can be read
        for name in self.follower_arms:
            self.follower_arms[name].read("Present_Position")
        for name in self.leader_arms:
            self.leader_arms[name].read("Present_Position")

        # Connect the cameras
        for name in self.cameras:
            try:
                self.cameras[name].connect()
            except Exception as e:
                print(f"Failed to connect camera {name}: {e}")
                continue

        self.is_connected = True

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("ManipulatorRobot is not connected. You need to run `robot.connect()`.")
        for name in self.follower_arms:
            self.follower_arms[name].disconnect()
        for name in self.leader_arms:
            self.leader_arms[name].disconnect()
        for name in self.cameras:
            self.cameras[name].disconnect()

        self.is_connected = False

    def teleop_step(self, record_data=False) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("ManipulatorRobot is not connected. You need to run `robot.connect()`.")
        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].read("Present_Position")
            leader_pos[name] = torch.from_numpy(leader_pos[name])
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t
