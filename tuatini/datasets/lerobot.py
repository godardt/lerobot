import json
import logging
import os
from pathlib import Path
from typing import Callable

import datasets
import packaging.version
import torch

from tuatini.datasets.compute_stats import aggregate_stats
from tuatini.devices.robots import Robot
from tuatini.utils.datasets import (
    get_safe_version,
    hf_transform_to_torch,
    hw_to_dataset_features,
    write_episode,
    write_episode_stats,
    write_info,
)
from tuatini.utils.image_writer import AsyncImageWriter
from tuatini.utils.time import check_delta_timestamps, check_timestamps_sync, get_delta_indices
from tuatini.utils.video import get_safe_default_codec

CODEBASE_VERSION = "v2.1"

default_home = os.path.join(os.path.expanduser("~"), ".cache")
HF_HOME = os.path.expandvars(
    os.path.expanduser(
        os.getenv(
            "HF_HOME",
            os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "huggingface"),
        )
    )
)
default_cache_path = Path(HF_HOME) / "lerobot"
HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", default_cache_path)).expanduser()

DEFAULT_CHUNK_SIZE = 1000  # Max number of episodes per chunk
DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
DEFAULT_PARQUET_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
INFO_PATH = "meta/info.json"


def create_empty_dataset_info(codebase_version: str, fps: int, robot_type: str, features: dict) -> dict:
    return {
        "codebase_version": codebase_version,
        "robot_type": robot_type,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": 0,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": fps,
        "splits": {},
        "data_path": DEFAULT_PARQUET_PATH,
        "video_path": DEFAULT_VIDEO_PATH,
        "features": features,
    }


class LeRobotDatasetMetadata:
    """
    https://github.com/huggingface/lerobot?tab=readme-ov-file#the-lerobotdataset-format
    """

    def __init__(
        self,
        repo_id: str,
        robot: Robot,
        fps: int,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
    ):
        self.repo_id = repo_id
        self.robot = robot
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        features = self.get_features_from_robot()
        robot_type = self.robot.type
        if not all(cam.fps == fps for cam in self.robot.cameras.values()):
            logging.warning(
                f"Some cameras in your {robot.robot_type} robot don't have an fps matching the fps of your dataset."
                "In this case, frames from lower fps cameras will be repeated to fill in the blanks."
            )

        self.tasks, self.task_to_task_index = {}, {}
        self.episodes_stats, self.stats, self.episodes = {}, {}, {}
        info = create_empty_dataset_info(CODEBASE_VERSION, fps, robot_type, features)
        fpath = root / INFO_PATH
        fpath.parent.mkdir(exist_ok=True, parents=True)
        with open(fpath, "w") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)

    def get_features_from_robot(self, use_videos: bool = True) -> dict:
        action_features = hw_to_dataset_features(self.robot.action_features, "action", use_video=use_videos)
        obs_features = hw_to_dataset_features(self.robot.observation_features, "observation", use_video=use_videos)
        dataset_features = {**action_features, **obs_features}
        return dataset_features

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1

        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)
        if len(self.video_keys) > 0:
            self.update_video_info()

        write_info(self.info, self.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)

        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        """
        2 modes are available for instantiating this class, depending on 2 different use cases:

        1. Your dataset already exists:
            - On your local disk in the 'root' folder. This is typically the case when you recorded your
              dataset locally and you may or may not have pushed it to the hub yet. Instantiating this class
              with 'root' will load your dataset directly from disk. This can happen while you're offline (no
              internet connection).

            - On the Hugging Face Hub at the address https://huggingface.co/datasets/{repo_id} and not on
              your local disk in the 'root' folder. Instantiating this class with this 'repo_id' will download
              the dataset from that address and load it, pending your dataset is compliant with
              codebase_version v2.0. If your dataset has been created before this new format, you will be
              prompted to convert it using our conversion script from v1.6 to v2.0, which you can find at
              lerobot/common/datasets/v2/convert_dataset_v1_to_v2.py.


        2. Your dataset doesn't already exists (either on local disk or on the Hub): you can create an empty
           LeRobotDataset with the 'create' classmethod. This can be used for recording a dataset or port an
           existing dataset to the LeRobotDataset format.


        In terms of files, LeRobotDataset encapsulates 3 main things:
            - metadata:
                - info contains various information about the dataset like shapes, keys, fps etc.
                - stats stores the dataset statistics of the different modalities for normalization
                - tasks contains the prompts for each task of the dataset, which can be used for
                  task-conditioned training.
            - hf_dataset (from datasets.Dataset), which will read any values from parquet files.
            - videos (optional) from which frames are loaded to be synchronous with data from parquet files.

        A typical LeRobotDataset looks like this from its root path:
        .
        ├── data
        │   ├── chunk-000
        │   │   ├── episode_000000.parquet
        │   │   ├── episode_000001.parquet
        │   │   ├── episode_000002.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── episode_001000.parquet
        │   │   ├── episode_001001.parquet
        │   │   ├── episode_001002.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes.jsonl
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.jsonl
        └── videos
            ├── chunk-000
            │   ├── observation.images.laptop
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            │   ├── observation.images.phone
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            ├── chunk-001
            └── ...

        Note that this file-based structure is designed to be as versatile as possible. The files are split by
        episodes which allows a more granular control over which episodes one wants to use and download. The
        structure of the dataset is entirely described in the info.json file, which can be easily downloaded
        or viewed directly on the hub before downloading any actual data. The type of files used are very
        simple and do not need complex tools to be read, it only uses .parquet, .json and .mp4 files (and .md
        for the README).

        Args:
            repo_id (str): This is the repo id that will be used to fetch the dataset. Locally, the dataset
                will be stored under root/repo_id.
            root (Path | None, optional): Local directory to use for downloading/writing files. You can also
                set the LEROBOT_HOME environment variable to point to a different location. Defaults to
                '~/.cache/huggingface/lerobot'.
            episodes (list[int] | None, optional): If specified, this will only load episodes specified by
                their episode_index in this list. Defaults to None.
            image_transforms (Callable | None, optional): You can pass standard v2 image transforms from
                torchvision.transforms.v2 here which will be applied to visual modalities (whether they come
                from videos or images). Defaults to None.
            delta_timestamps (dict[list[float]] | None, optional): _description_. Defaults to None.
            tolerance_s (float, optional): Tolerance in seconds used to ensure data timestamps are actually in
                sync with the fps value. It is used at the init of the dataset to make sure that each
                timestamps is separated to the next by 1/fps +/- tolerance_s. This also applies to frames
                decoded from video files. It is also used to check that `delta_timestamps` (when provided) are
                multiples of 1/fps. Defaults to 1e-4.
            revision (str, optional): An optional Git revision id which can be a branch name, a tag, or a
                commit hash. Defaults to current codebase version tag.
            sync_cache_first (bool, optional): Flag to sync and refresh local files first. If True and files
                are already present in the local cache, this will be faster. However, files loaded might not
                be in sync with the version on the hub, especially if you specified 'revision'. Defaults to
                False.
            download_videos (bool, optional): Flag to download the videos. Note that when set to True but the
                video files are already present on local disk, they won't be downloaded again. Defaults to
                True.
            video_backend (str | None, optional): Video backend to use for decoding videos. Defaults to torchcodec when available int the platform; otherwise, defaults to 'pyav'.
                You can also use the 'pyav' decoder used by Torchvision, which used to be the default option, or 'video_reader' which is another decoder of Torchvision.
        """
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotDatasetMetadata(self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync)
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)

        # Load actual data
        try:
            if force_cache_sync:
                raise FileNotFoundError
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = LeRobotDataset.get_episode_data_index(self.meta.episodes, self.episodes)

        # Check timestamps
        timestamps = torch.stack(self.hf_dataset["timestamp"]).numpy()
        episode_indices = torch.stack(self.hf_dataset["episode_index"]).numpy()
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def start_image_writer(self, num_processes: int = 0, num_threads: int = 4) -> None:
        if isinstance(self.image_writer, AsyncImageWriter):
            logging.warning(
                "You are starting a new AsyncImageWriter that is replacing an already existing one in the dataset."
            )

        self.image_writer = AsyncImageWriter(
            num_processes=num_processes,
            num_threads=num_threads,
        )

    def create_episode_buffer(self, episode_index: int | None = None) -> dict:
        current_ep_idx = self.meta.total_episodes if episode_index is None else episode_index
        ep_buffer = {}
        # size and task are special cases that are not in self.features
        ep_buffer["size"] = 0
        ep_buffer["task"] = []
        for key in self.features:
            ep_buffer[key] = current_ep_idx if key == "episode_index" else []
        return ep_buffer

    def get_hf_features_from_features(self) -> datasets.Features:
        hf_features = {}
        for key, ft in self.features.items():
            if ft["dtype"] == "video":
                continue
            elif ft["dtype"] == "image":
                hf_features[key] = datasets.Image()
            elif ft["shape"] == (1,):
                hf_features[key] = datasets.Value(dtype=ft["dtype"])
            elif len(ft["shape"]) == 1:
                hf_features[key] = datasets.Sequence(length=ft["shape"][0], feature=datasets.Value(dtype=ft["dtype"]))
            elif len(ft["shape"]) == 2:
                hf_features[key] = datasets.Array2D(shape=ft["shape"], dtype=ft["dtype"])
            elif len(ft["shape"]) == 3:
                hf_features[key] = datasets.Array3D(shape=ft["shape"], dtype=ft["dtype"])
            elif len(ft["shape"]) == 4:
                hf_features[key] = datasets.Array4D(shape=ft["shape"], dtype=ft["dtype"])
            elif len(ft["shape"]) == 5:
                hf_features[key] = datasets.Array5D(shape=ft["shape"], dtype=ft["dtype"])
            else:
                raise ValueError(f"Corresponding feature is not valid: {ft}")

        return datasets.Features(hf_features)

    def set_data_format(self, format: str) -> None:
        if format == "torch":
            self.hf_dataset.set_transform(hf_transform_to_torch)
        else:
            raise ValueError(f"Invalid format: {format}")

    def create_hf_dataset(self) -> datasets.Dataset:
        features = self.get_hf_features_from_features()
        ft_dict = {col: [] for col in features}
        hf_dataset = datasets.Dataset.from_dict(ft_dict, features=features, split="train")

        hf_dataset.set_format("torch")
        return hf_dataset

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        root: str | Path | None = None,
        robot: Robot | None = None,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
    ) -> "LeRobotDataset":
        """
        Create a LeRobot Dataset from scratch in order to record data.
        Args:
            repo_id (str): The repo id of the dataset.
            fps (int): The fps of the dataset.
            root (str | Path | None): The root of the dataset.
            robot (Robot | None): The robot to record data from.
            robot_type (str | None): The type of the robot.
            features (dict | None): The features of the dataset.
            tolerance_s (float): The tolerance of the dataset.
            image_writer_processes (int): Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
                set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
                and threads depends on your system. We recommend 4 threads per camera with 0 processes.
                If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
            image_writer_threads (int): Number of threads per subprocess.
            video_backend (str | None): The video backend to use for the dataset.
        """
        obj = cls.__new__(cls)
        obj.meta = LeRobotDatasetMetadata(
            repo_id=repo_id,
            robot=robot,
            fps=fps,
            root=root,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        # TODO: Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        return obj
