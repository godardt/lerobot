import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import rerun as rr
from termcolor import colored

root_dir = Path(__file__).parent.parent


def init_logging():
    def custom_format(record):
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fnameline = f"{record.pathname}:{record.lineno}"
        message = f"{record.levelname} {dt} {fnameline[-15:]:>15} {record.msg}"
        return message

    logging.basicConfig(level=logging.INFO)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    formatter = logging.Formatter()
    formatter.format = custom_format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)


def log_control_info(dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1 / dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    info_str = " ".join(log_items)
    logging.info(info_str)


def log_rr_event(action, observation):
    if action is not None:
        for k, v in action.items():
            for i, vv in enumerate(v):
                rr.log(f"sent_{k}_{i}", rr.Scalars(vv.numpy()))

    image_keys = [key for key in observation if "image" in key]
    for key in image_keys:
        rr.log(key, rr.Image(observation[key].numpy()), static=True)


def init_rerun(session_name: str = "lerobot_control_loop") -> None:
    """Initializes the Rerun SDK for visualizing the control loop.

    Args:
        control_config: Configuration determining data display and robot type.
        session_name: Rerun session name. Defaults to "lerobot_control_loop".

    Raises:
        ValueError: If viewer IP is missing for non-remote configurations with display enabled.
    """
    # Configure Rerun flush batch size default to 8KB if not set
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size

    # Initialize Rerun based on configuration
    rr.init(session_name)

    # Use a fixed temporary directory name
    temp_dir = Path(tempfile.gettempdir()) / "lerobot_rerun"
    rrd_file = temp_dir / "rerun" / f"{session_name}.rrd"
    rrd_file.parent.mkdir(parents=True, exist_ok=True)
    rr.save(rrd_file)
    print(f"Rerun initialized. The logs files are stored into {rrd_file}")
