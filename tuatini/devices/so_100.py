import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from tuatini.devices.camera import Camera, IPCamera, OpenCVCamera
from tuatini.devices.robots import Robot
from tuatini.motors.feetch import FeetechMotorsBus, FeetechTorqueMode
from tuatini.utils.exceptions import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from tuatini.utils.io import get_arm_id, substitute_path_variables


class SO100Robot(Robot):
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
        self.logs = {}

        # TODO: Add this inside this class (not the config)
        # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
        # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
        # the number of motors in your follower arms.
        self.max_relative_target: int | None = None

    @property
    def type(self):
        return "so_100"

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def cameras(self) -> dict[str, Camera]:
        return self.cameras

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
            rotation = cam_config["rotation"]

            if cam_config["type"] == "ip_camera":
                ip = cam_config["vcam_ip"]
                port = cam_config["vcam_port"]
                cameras[cam_name] = IPCamera(ip, port, fps, width, height, rotation)
            elif cam_config["type"] == "opencv":
                device = cam_config["device"]
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
        unavailable_cameras = []
        for name in self.cameras:
            try:
                self.cameras[name].connect()
            except Exception as e:
                print(f"Failed to connect camera {name}: {e}")
                unavailable_cameras.append(name)
                continue

        # Remove the camera from the list of cameras
        for name in unavailable_cameras:
            del self.cameras[name]

        logging.info(f"Connected {len(self.cameras)} cameras: {list(self.cameras.keys())}")
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

    def _ensure_safe_goal_position(
        goal_pos: torch.Tensor, present_pos: torch.Tensor, max_relative_target: float | list[float]
    ):
        # Cap relative action target magnitude for safety.
        diff = goal_pos - present_pos
        max_relative_target = torch.tensor(max_relative_target)
        safe_diff = torch.minimum(diff, max_relative_target)
        safe_diff = torch.maximum(safe_diff, -max_relative_target)
        safe_goal_pos = present_pos + safe_diff

        if not torch.allclose(goal_pos, safe_goal_pos):
            logging.warning(
                "Relative goal position magnitude had to be clamped to be safe.\n"
                f"  requested relative goal position target: {diff}\n"
                f"    clamped relative goal position target: {safe_diff}"
            )

        return safe_goal_pos

    def teleop_step(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("ManipulatorRobot is not connected. You need to run `robot.connect()`.")
        leader_pos = {}

        # Prepare to assign the position of the leader to the follower
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].read("Present_Position")
            leader_pos[name] = torch.from_numpy(leader_pos[name])
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t

        # Send goal position to the follower
        follower_goal_pos = {}
        for name in self.follower_arms:
            before_fwrite_t = time.perf_counter()
            goal_pos = leader_pos[name]

            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            if self.max_relative_target is not None:
                present_pos = self.follower_arms[name].read("Present_Position")
                present_pos = torch.from_numpy(present_pos)
                goal_pos = self._ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            # Used when record_data=True
            follower_goal_pos[name] = goal_pos

            goal_pos = goal_pos.numpy().astype(np.float32)
            self.follower_arms[name].write("Goal_Position", goal_pos)
            self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - before_fwrite_t

        # TODO: Add velocity and other info
        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        # Create action by concatenating follower goal position
        action = []
        for name in self.follower_arms:
            if name in follower_goal_pos:
                action.append(follower_goal_pos[name])
        action = torch.cat(action)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict
