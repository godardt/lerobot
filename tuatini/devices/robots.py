from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from tuatini.devices.camera import Camera


class MotorNormMode(str, Enum):
    RANGE_0_100 = "range_0_100"
    RANGE_M100_100 = "range_m100_100"
    DEGREES = "degrees"


@dataclass
class Motor:
    id: int
    model: str
    norm_mode: MotorNormMode


@dataclass
class MotorCalibration:
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


class Robot(Protocol):
    robot_type: str
    features: dict

    @property
    def type(self) -> int: ...

    @property
    def num_cameras(self) -> int: ...

    @property
    def cameras(self) -> dict[str, Camera]: ...

    @property
    def action_features(self) -> dict[str, type]: ...

    @property
    def observation_features(self) -> dict[str, type]: ...

    @property
    def is_calibrated(self) -> bool: ...

    def connect(self): ...
    def disconnect(self): ...
    def calibrate(self): ...
    def send_action(self, action): ...
