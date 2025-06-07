import time
from pathlib import Path

import click
import yaml

from tuatini.datasets.lerobot import LeRobotDataset
from tuatini.devices.so_100 import SO100Robot
from tuatini.utils.logs import init_logging, init_rerun, log_control_info, log_rr_event
from tuatini.utils.time import busy_wait

root_dir = Path(__file__).parent.parent


def safe_disconnect(func):
    def wrapper(robot, *args, **kwargs):
        try:
            return func(robot, *args, **kwargs)
        except Exception as e:
            if robot.is_connected:
                robot.disconnect()
            raise e

    return wrapper


def sanity_check_dataset_name(repo_id, policy_cfg):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided ({policy_cfg.type})."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


@safe_disconnect
def _record_dataset(
    leader_robots: list[SO100Robot],
    follower_robots: list[SO100Robot],
    dataset_recording_config: dict,
    record_data=False,
    control_time_s=float("inf"),
    single_task: None | str = None,
):
    sanity_check_dataset_name(dataset_recording_config["repo_id"], dataset_recording_config.get("policy"))
    dataset = LeRobotDataset.create(
        dataset_recording_config["repo_id"],
        fps=dataset_recording_config["fps"],
        robot=robot,
        image_writer_processes=1,
        image_writer_threads=len(robot.cameras),
    )

    if not robot.is_connected:
        robot.connect()

    timestamp = 0
    start_episode_t = time.perf_counter()
    fps = dataset_recording_config.get("fps", 30)

    for i in reversed(range(3)):
        print(f"Teleoperating robot in {i + 1} seconds")
        time.sleep(1)

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        observation, action = robot.teleop_step(record_data=record_data)
        if dataset is not None:
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)

        log_rr_event(action, observation)
        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        log_control_info(dt_s, fps=fps)
        timestamp = time.perf_counter() - start_episode_t


@click.command("Straightforward way to teleoperate the SO-100 robot")
@click.option(
    "--config", type=str, help="Config file for the robot", default=str(root_dir / "config" / "SO-100_ROG.yaml")
)
def main(config):
    init_logging()
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    leader_robots = []
    follower_robots = []

    for robot_type in ["leader_arms", "follower_arms"]:
        for arm_config in config["robots"][robot_type]:
            robot = SO100Robot(arm_config)
            robot.connect()
            if robot_type == "leader_arms":
                leader_robots.append(robot)
            else:
                follower_robots.append(robot)

    init_rerun()

    _record_dataset(leader_robots, follower_robots, config.get("dataset_recording"), record_data=True)
    print("Shutting down...")


if __name__ == "__main__":
    main()
