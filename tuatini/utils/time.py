import platform
import time
from datetime import datetime, timezone
from pprint import pformat

import numpy as np


def capture_timestamp_utc():
    return datetime.now(timezone.utc)


def busy_wait(seconds):
    if platform.system() == "Darwin":
        # On Mac, `time.sleep` is not accurate and we need to use this while loop trick,
        # but it consumes CPU cycles.
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            pass
    else:
        # On Linux time.sleep is accurate
        if seconds > 0:
            time.sleep(seconds)


def check_timestamps_sync(
    timestamps: np.ndarray,
    episode_indices: np.ndarray,
    episode_data_index: dict[str, np.ndarray],
    fps: int,
    tolerance_s: float,
    raise_value_error: bool = True,
) -> bool:
    """
    This check is to make sure that each timestamp is separated from the next by (1/fps) +/- tolerance
    to account for possible numerical error.

    Args:
        timestamps (np.ndarray): Array of timestamps in seconds.
        episode_indices (np.ndarray): Array indicating the episode index for each timestamp.
        episode_data_index (dict[str, np.ndarray]): A dictionary that includes 'to',
            which identifies indices for the end of each episode.
        fps (int): Frames per second. Used to check the expected difference between consecutive timestamps.
        tolerance_s (float): Allowed deviation from the expected (1/fps) difference.
        raise_value_error (bool): Whether to raise a ValueError if the check fails.

    Returns:
        bool: True if all checked timestamp differences lie within tolerance, False otherwise.

    Raises:
        ValueError: If the check fails and `raise_value_error` is True.
    """
    if timestamps.shape != episode_indices.shape:
        raise ValueError(
            "timestamps and episode_indices should have the same shape. "
            f"Found {timestamps.shape=} and {episode_indices.shape=}."
        )

    # Consecutive differences
    diffs = np.diff(timestamps)
    within_tolerance = np.abs(diffs - (1.0 / fps)) <= tolerance_s

    # Mask to ignore differences at the boundaries between episodes
    mask = np.ones(len(diffs), dtype=bool)
    ignored_diffs = episode_data_index["to"][:-1] - 1  # indices at the end of each episode
    mask[ignored_diffs] = False
    filtered_within_tolerance = within_tolerance[mask]

    # Check if all remaining diffs are within tolerance
    if not np.all(filtered_within_tolerance):
        # Track original indices before masking
        original_indices = np.arange(len(diffs))
        filtered_indices = original_indices[mask]
        outside_tolerance_filtered_indices = np.nonzero(~filtered_within_tolerance)[0]
        outside_tolerance_indices = filtered_indices[outside_tolerance_filtered_indices]

        outside_tolerances = []
        for idx in outside_tolerance_indices:
            entry = {
                "timestamps": [timestamps[idx], timestamps[idx + 1]],
                "diff": diffs[idx],
                "episode_index": episode_indices[idx].item()
                if hasattr(episode_indices[idx], "item")
                else episode_indices[idx],
            }
            outside_tolerances.append(entry)

        if raise_value_error:
            raise ValueError(
                f"""One or several timestamps unexpectedly violate the tolerance inside episode range.
                This might be due to synchronization issues during data collection.
                \n{pformat(outside_tolerances)}"""
            )
        return False

    return True


def check_delta_timestamps(
    delta_timestamps: dict[str, list[float]], fps: int, tolerance_s: float, raise_value_error: bool = True
) -> bool:
    """This will check if all the values in delta_timestamps are multiples of 1/fps +/- tolerance.
    This is to ensure that these delta_timestamps added to any timestamp from a dataset will themselves be
    actual timestamps from the dataset.
    """
    outside_tolerance = {}
    for key, delta_ts in delta_timestamps.items():
        within_tolerance = [abs(ts * fps - round(ts * fps)) / fps <= tolerance_s for ts in delta_ts]
        if not all(within_tolerance):
            outside_tolerance[key] = [
                ts for ts, is_within in zip(delta_ts, within_tolerance, strict=True) if not is_within
            ]

    if len(outside_tolerance) > 0:
        if raise_value_error:
            raise ValueError(
                f"""
                The following delta_timestamps are found outside of tolerance range.
                Please make sure they are multiples of 1/{fps} +/- tolerance and adjust
                their values accordingly.
                \n{pformat(outside_tolerance)}
                """
            )
        return False

    return True


def get_delta_indices(delta_timestamps: dict[str, list[float]], fps: int) -> dict[str, list[int]]:
    delta_indices = {}
    for key, delta_ts in delta_timestamps.items():
        delta_indices[key] = [round(d * fps) for d in delta_ts]

    return delta_indices
