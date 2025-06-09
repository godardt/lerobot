from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent


def get_arm_id(name, arm_type):
    """Returns the string identifier of a robot arm. For instance, for a bimanual manipulator
    like Aloha, it could be left_follower, right_follower, left_leader, or right_leader.
    """
    return f"{name}_{arm_type}"


def substitute_path_variables(path):
    path = path.replace("$PROJECT_DIR", str(PROJECT_DIR))
    path = path.replace("$DATETIME", datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    return path
