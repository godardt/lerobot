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


class FeetechMotorsBus:
    """
    The FeetechMotorsBus class allows to efficiently read and write to the attached motors. It relies on
    the python [feetech sdk](https://github.com/ftservo/FTServo_Python) available on [Pypi](https://pypi.org/project/ftservo-python-sdk/).

    A FeetechMotorsBus instance requires a port (e.g. `FeetechMotorsBus(port="/dev/tty.usbmodem575E0031751"`)).
    """

    def __init__(self, motors, port, mock=False):
        self.motors = motors
        self.port = port
        self.mock = mock

        self.model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
        self.model_resolution = deepcopy(MODEL_RESOLUTION)

        self.port_handler = None
        self.packet_handler = None
        self.calibration = None
        self.is_connected = False
        self.group_readers = {}
        self.group_writers = {}
        self.logs = {}

        self.track_positions = {}

    def connect(self):
        pass

    def disconnect(self):
        pass

    def read(self):
        pass
