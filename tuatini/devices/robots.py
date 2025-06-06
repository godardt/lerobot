from typing import Protocol

from tuatini.devices.cameras import Camera


class Robot(Protocol):
    robot_type: str
    features: dict

    @property
    def type(self) -> int:
        pass

    @property
    def num_cameras(self) -> int:
        pass

    @property
    def cameras(self) -> dict[str, Camera]:
        pass

    def connect(self): ...
    def run_calibration(self): ...
    def teleop_step(self, record_data=False): ...
    def capture_observation(self): ...
    def send_action(self, action): ...
    def disconnect(self): ...
