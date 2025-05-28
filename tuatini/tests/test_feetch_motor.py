from tuatini.motors.feetch import SCS_SERIES_CONTROL_TABLE, FeetechMotorsBus


def test_feetch_motor_buses_read():
    follower_arm_device = "/dev/ttyACM1"
    motors_bus_configs = {
        "shoulder_pan": (1, "sts3215"),
        "shoulder_lift": (2, "sts3215"),
        "elbow_flex": (3, "sts3215"),
        "wrist_flex": (4, "sts3215"),
        "wrist_roll": (5, "sts3215"),
        "gripper": (6, "sts3215"),
    }
    motors_bus = FeetechMotorsBus(motors_bus_configs, port=follower_arm_device)
    motors_bus.connect()
    for motor_index_name in SCS_SERIES_CONTROL_TABLE:
        val = motors_bus.read(motor_index_name)
        assert val is not None
        print(f"{motor_index_name}: {val}")
    motors_bus.disconnect()


def test_feetch_motor_buses_write():
    follower_arm_device = "/dev/ttyACM1"
    motors_bus_configs = {
        "shoulder_pan": (1, "sts3215"),
        "shoulder_lift": (2, "sts3215"),
        "elbow_flex": (3, "sts3215"),
        "wrist_flex": (4, "sts3215"),
        "wrist_roll": (5, "sts3215"),
        "gripper": (6, "sts3215"),
    }
    motors_bus = FeetechMotorsBus(motors_bus_configs, port=follower_arm_device)
    motors_bus.connect()
    current_position = motors_bus.read("Present_Position")
    few_steps = 30
    motors_bus.write("Goal_Position", current_position + few_steps)
    new_position = motors_bus.read("Goal_Position")
    assert all(new_position == current_position + few_steps)
    motors_bus.disconnect()
