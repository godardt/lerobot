from tuatini.motors.feetch import FeetechMotorsBus


def test_feetch_motor_bus():
    motors_bus = FeetechMotorsBus(port="/dev/ttyUSB0")
    motors_bus.connect()
    motors_bus.disconnect()
