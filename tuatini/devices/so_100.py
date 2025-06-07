import json
import logging
from functools import cached_property
from pathlib import Path
from pprint import pformat
from typing import Any

from tuatini.devices.camera import Camera
from tuatini.devices.robots import Motor, MotorCalibration, MotorNormMode, Robot
from tuatini.motors.feetech import FeetechMotorsBus, OperatingMode
from tuatini.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from tuatini.utils.io import substitute_path_variables


class SO100Robot(Robot):
    def __init__(self, device, calibration_dir, cameras=None):
        norm_mode_body = MotorNormMode.RANGE_M100_100
        calibration_dir = Path(substitute_path_variables(calibration_dir))
        self.calibration: dict[str, MotorCalibration] = self._load_calibration(calibration_dir)
        self.bus = FeetechMotorsBus(
            port=device,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self._cameras = cameras if cameras else {}

    def _load_calibration(self, fpath: Path) -> None:
        if not fpath.exists():
            return {}

        with open(fpath) as f:
            raw_calibration = json.load(f)
            self.calibration = {
                motor_name: MotorCalibration(
                    id=cal["id"],
                    drive_mode=cal["drive_mode"],
                    homing_offset=cal["homing_offset"],
                    range_min=cal["range_min"],
                    range_max=cal["range_max"],
                )
                for motor_name, cal in raw_calibration.items()
            }

    def _save_calibration(self, fpath: Path) -> None:
        with open(fpath, "w") as f:
            calibration_dict = {
                motor_name: {
                    "id": cal.id,
                    "drive_mode": cal.drive_mode,
                    "homing_offset": cal.homing_offset,
                    "range_min": cal.range_min,
                    "range_max": cal.range_max,
                }
                for motor_name, cal in self.calibration.items()
            }
            json.dump(calibration_dict, f, indent=4)

    def calibrate(self) -> None:
        logging.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write_register("Operating_Mode", motor, OperatingMode.POSITION.value)

        logging.info(
            "Calibration guide: https://github.com/huggingface/lerobot/blob/main/lerobot/common/robots/so100_follower/so100.mdx#calibrate"
        )
        logging.info("Calibration video: https://huggingface.co/docs/lerobot/en/so101#calibration-video")
        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in self.bus.motors if motor != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self._cameras.values())

    @property
    def type(self):
        return "so_100"

    @property
    def num_cameras(self):
        return len(self._cameras)

    @property
    def cameras(self) -> dict[str, Camera]:
        return self._cameras

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write_register("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
                self.bus.write_register("P_Coefficient", motor, 16)
                # Set I_Coefficient and D_Coefficient to default value 0 and 32
                self.bus.write_register("I_Coefficient", motor, 0)
                self.bus.write_register("D_Coefficient", motor, 4)

    def connect(self, calibrate: bool = True):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        # Connect the cameras
        unavailable_cameras = []
        for cam_name in self._cameras.values():
            try:
                self._cameras[cam_name].connect()
            except Exception as e:
                print(f"Failed to connect camera {cam_name}: {e}")
                unavailable_cameras.append(cam_name)
                continue

        # Remove the camera from the list of cameras
        for name in unavailable_cameras:
            del self._cameras[name]

        self.configure()
        logging.info(f"{self} connected with {len(self._cameras)} cameras: {list(self._cameras.keys())}")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logging.info(f"{self} disconnected.")

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = self._ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def _ensure_safe_goal_position(
        self, goal_present_pos: dict[str, tuple[float, float]], max_relative_target: float | dict[float]
    ) -> dict[str, float]:
        """Caps relative action target magnitude for safety."""

        if isinstance(max_relative_target, float):
            diff_cap = dict.fromkeys(goal_present_pos, max_relative_target)
        elif isinstance(max_relative_target, dict):
            if not set(goal_present_pos) == set(max_relative_target):
                raise ValueError("max_relative_target keys must match those of goal_present_pos.")
            diff_cap = max_relative_target
        else:
            raise TypeError(max_relative_target)

        warnings_dict = {}
        safe_goal_positions = {}
        for key, (goal_pos, present_pos) in goal_present_pos.items():
            diff = goal_pos - present_pos
            max_diff = diff_cap[key]
            safe_diff = min(diff, max_diff)
            safe_diff = max(safe_diff, -max_diff)
            safe_goal_pos = present_pos + safe_diff
            safe_goal_positions[key] = safe_goal_pos
            if abs(safe_goal_pos - goal_pos) > 1e-4:
                warnings_dict[key] = {
                    "original goal_pos": goal_pos,
                    "safe goal_pos": safe_goal_pos,
                }

        if warnings_dict:
            logging.warning(
                f"Relative goal position magnitude had to be clamped to be safe.\n{pformat(warnings_dict, indent=4)}"
            )

        return safe_goal_positions

    # def teleop_step(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    #     if not self.is_connected:
    #         raise RobotDeviceNotConnectedError("ManipulatorRobot is not connected. You need to run `robot.connect()`.")
    #     leader_pos = {}

    #     # Prepare to assign the position of the leader to the follower
    #     for name in self.leader_arms:
    #         before_lread_t = time.perf_counter()
    #         leader_pos[name] = self.leader_arms[name].read("Present_Position")
    #         leader_pos[name] = torch.from_numpy(leader_pos[name])
    #         self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t

    #     # Send goal position to the follower
    #     follower_goal_pos = {}
    #     for name in self.follower_arms:
    #         before_fwrite_t = time.perf_counter()
    #         goal_pos = leader_pos[name]

    #         # Cap goal position when too far away from present position.
    #         # Slower fps expected due to reading from the follower.
    #         if self.max_relative_target is not None:
    #             present_pos = self.follower_arms[name].read("Present_Position")
    #             present_pos = torch.from_numpy(present_pos)
    #             goal_pos = self._ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

    #         # Used when record_data=True
    #         follower_goal_pos[name] = goal_pos

    #         goal_pos = goal_pos.numpy().astype(np.float32)
    #         self.follower_arms[name].write("Goal_Position", goal_pos)
    #         self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - before_fwrite_t

    #     # TODO: Add velocity and other info
    #     # Read follower position
    #     follower_pos = {}
    #     for name in self.follower_arms:
    #         before_fread_t = time.perf_counter()
    #         follower_pos[name] = self.follower_arms[name].read("Present_Position")
    #         follower_pos[name] = torch.from_numpy(follower_pos[name])
    #         self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

    #     # Create state by concatenating follower current position
    #     state = []
    #     for name in self.follower_arms:
    #         if name in follower_pos:
    #             state.append(follower_pos[name])
    #     state = torch.cat(state)

    #     # Create action by concatenating follower goal position
    #     action = []
    #     for name in self.follower_arms:
    #         if name in follower_goal_pos:
    #             action.append(follower_goal_pos[name])
    #     action = torch.cat(action)

    #     # Capture images from cameras
    #     images = {}
    #     for name in self._cameras:
    #         before_camread_t = time.perf_counter()
    #         images[name] = self._cameras[name].async_read()
    #         images[name] = torch.from_numpy(images[name])
    #         self.logs[f"read_camera_{name}_dt_s"] = self._cameras[name].logs["delta_timestamp_s"]
    #         self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

    #     # Populate output dictionaries
    #     obs_dict, action_dict = {}, {}
    #     obs_dict["observation.state"] = state
    #     action_dict["action"] = action
    #     for name in self._cameras:
    #         obs_dict[f"observation.images.{name}"] = images[name]

    #     return obs_dict, action_dict
