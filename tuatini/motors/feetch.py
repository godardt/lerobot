import scservo_sdk as scs

PROTOCOL_VERSION = 0
TIMEOUT_MS = 1000


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

    def __init__(self, motors, port):
        self.motors = motors
        self.port = port

        self.port_handler = None
        self.packet_handler = None
        self.group_readers = {}
        self.group_writers = {}
        self.is_connected = False

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"FeetechMotorsBus({self.port}) is already connected. Do not call `motors_bus.connect()` twice."
            )
        self.port_handler = scs.PortHandler(self.port)
        self.packet_handler = scs.PacketHandler(PROTOCOL_VERSION)

        if not self.port_handler.openPort():
            raise OSError(f"Failed to open port '{self.port}'.")

        # Allow to read and write
        self.is_connected = True

        self.port_handler.setPacketTimeoutMillis(TIMEOUT_MS)

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"FeetechMotorsBus({self.port}) is not connected. Try running `motors_bus.connect()` first."
            )

        if self.port_handler is not None:
            self.port_handler.closePort()
            self.port_handler = None

        self.packet_handler = None
        self.group_readers = {}
        self.group_writers = {}
        self.is_connected = False

    def read(self):
        pass
