# Method 4: Multi-Task Mixed Data Loading
#
# Mixes emg_nature (gesture labels) and emg2pose (joint angle labels) data.

from pathlib import Path
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
import pytorch_lightning as pl


class Emg2PoseJointAngleDataset(Dataset):
    """
    Dataset returning EMG + joint angle windows from emg2pose HDF5 files.

    For each window, also generates weak gesture labels from joint angles
    when weak_gesture_labels=True.
    """

    def __init__(
        self,
        hdf5_paths: list[Path],
        window_length: int = 16000,
        stride: int = 16000,
        max_windows_per_file: int = 30,
        weak_gesture_labels: bool = True,
    ):
        self.hdf5_paths = hdf5_paths
        self.window_length = window_length
        self.stride = stride
        self.max_windows_per_file = max_windows_per_file
        self.weak_gesture_labels = weak_gesture_labels

        # Pre-compute windows
        self.windows: list[tuple[int, int, int]] = []
        for i, path in enumerate(hdf5_paths):
            try:
                with h5py.File(path, 'r') as f:
                    if 'emg2pose' in f:
                        total_samples = f['emg2pose']['timeseries'].shape[0]
                    elif 'timeseries' in f:
                        total_samples = f['timeseries'].shape[0]
                    else:
                        continue
                    if total_samples < window_length:
                        continue
                    num_windows = min(
                        (total_samples - window_length) // stride + 1,
                        max_windows_per_file,
                    )
                    for w in range(num_windows):
                        self.windows.append((i, w * stride, w * stride + window_length))
            except Exception:
                continue

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        file_idx, start, end = self.windows[idx]
        path = self.hdf5_paths[file_idx]

        with h5py.File(path, 'r') as f:
            if 'emg2pose' in f:
                ts = f['emg2pose']['timeseries']
            elif 'timeseries' in f:
                ts = f['timeseries']
            else:
                raise KeyError(f"No timeseries data in {path}")
            emg = ts['emg'][start:end].astype(np.float32)  # (T, 16)
            try:
                joint_angles = ts['joint_angles'][start:end].astype(np.float32)  # (T, 20)
            except (ValueError, KeyError):
                joint_angles = np.zeros((end - start, 20), dtype=np.float32)

        result = {
            "emg": torch.from_numpy(emg).permute(1, 0),  # (16, T)
            "joint_angles": torch.from_numpy(joint_angles).permute(1, 0),  # (20, T)
            "task": "joint",  # primary task is joint angle regression
        }

        if self.weak_gesture_labels:
            from emg_transfer.multitask_networks import JointAngleToGestureMapper
            ja = torch.from_numpy(joint_angles).unsqueeze(0).permute(0, 2, 1)  # (1, 20, T)
            weak_targets, confidence = JointAngleToGestureMapper.infer_gesture_from_angles(ja)
            result["weak_targets"] = weak_targets.squeeze(0)  # (9, T)
            result["weak_confidence"] = confidence.squeeze(0)  # (9, T)

        return result


class MixedEmgDataModule(pl.LightningDataModule):
    """
    DataModule that provides both gesture-labeled (emg_nature) and
    joint-angle-labeled (emg2pose) data batches.

    Training alternates between the two data sources.
    """

    def __init__(
        self,
        gesture_data_location: str,
        joint_data_location: str,
        window_length: int = 16000,
        stride: int = 16000,
        batch_size: int = 4,
        joint_batch_size: int = 4,
        num_workers: int = 0,
        joint_data_fraction: float = 0.5,  # fraction of batches that are joint data
        gesture_split_csv: str | None = None,
    ):
        super().__init__()
        self.gesture_data_location = gesture_data_location
        self.joint_data_location = joint_data_location
        self.window_length = window_length
        self.stride = stride
        self.batch_size = batch_size
        self.joint_batch_size = joint_batch_size
        self.num_workers = num_workers
        self.joint_data_fraction = joint_data_fraction
        self.gesture_split_csv = gesture_split_csv

    def setup(self, stage: str | None = None):
        # Gesture data: use existing emg_transfer pipeline
        from emg_transfer.data import DataSplit, make_dataset
        from emg_transfer.transforms import DiscreteGesturesTransform
        from emg_transfer.augmentation import RotationAugmentation

        if self.gesture_split_csv:
            data_split = DataSplit.from_csv(self.gesture_split_csv)
        else:
            # Default: use the standard split
            from hydra.utils import instantiate
            data_split = DataSplit.from_csv(
                f"{self.gesture_data_location}/discrete_gestures_corpus.csv"
            )

        transform = DiscreteGesturesTransform(pulse_window=[0.08, 0.12])
        augmentation = RotationAugmentation(rotation=2)

        if stage == "fit" or stage is None:
            self.train_gesture_dataset = make_dataset(
                data_location=self.gesture_data_location,
                transform=transform,
                partition_dict=data_split.train,
                window_length=self.window_length,
                stride=self.stride,
                jitter=True,
                emg_augmentation=augmentation,
                split_label="train_gesture",
            )

        if stage == "fit" or stage == "validate" or stage is None:
            self.val_gesture_dataset = make_dataset(
                data_location=self.gesture_data_location,
                transform=transform,
                partition_dict=data_split.val,
                window_length=self.window_length,
                stride=self.stride,
                jitter=False,
                emg_augmentation=None,
                split_label="val_gesture",
            )

        if stage == "test" or stage is None:
            self.test_gesture_dataset = make_dataset(
                data_location=self.gesture_data_location,
                transform=transform,
                partition_dict=data_split.test,
                window_length=None,
                stride=None,
                jitter=False,
                emg_augmentation=None,
                split_label="test_gesture",
            )

        # Joint angle data
        if stage == "fit" or stage is None:
            joint_dir = Path(self.joint_data_location)
            if not joint_dir.exists():
                joint_dir = Path("D:/emg/emg2pose1/emg2pose-main/emg2pose_dataset/emg2pose_data")
            hdf5_files = sorted(joint_dir.glob("*.hdf5"))
            # Use a subset to match training throughput
            n_files = min(len(hdf5_files), 100)
            hdf5_files = hdf5_files[:n_files]

            self.train_joint_dataset = Emg2PoseJointAngleDataset(
                hdf5_paths=hdf5_files[:int(n_files * 0.9)],
                window_length=self.window_length,
                stride=self.stride,
                max_windows_per_file=20,
                weak_gesture_labels=True,
            )

            self.val_joint_dataset = Emg2PoseJointAngleDataset(
                hdf5_paths=hdf5_files[int(n_files * 0.9):],
                window_length=self.window_length,
                stride=self.stride,
                max_windows_per_file=5,
                weak_gesture_labels=True,
            )

    def train_dataloader(self):
        gesture_loader = DataLoader(
            self.train_gesture_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=True,
        )
        joint_loader = DataLoader(
            self.train_joint_dataset, batch_size=self.joint_batch_size,
            num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=True,
        )

        return MixedDataLoader(gesture_loader, joint_loader, self.joint_data_fraction)

    def val_dataloader(self):
        return DataLoader(
            self.val_gesture_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=True, shuffle=False,
        )

    def test_dataloader(self):
        from emg_transfer.data_module import custom_collate_fn
        return DataLoader(
            self.test_gesture_dataset, batch_size=1,
            num_workers=self.num_workers, pin_memory=True, shuffle=False,
            collate_fn=custom_collate_fn,
        )


class MixedDataLoader:
    """
    Alternates between two DataLoaders.
    Yields (batch, task_type) where task_type is 'gesture' or 'joint'.
    """

    def __init__(self, loader_a, loader_b, fraction_b=0.5):
        self.loader_a = loader_a  # gesture
        self.loader_b = loader_b  # joint
        self.fraction_b = fraction_b

    def __iter__(self):
        iter_a = iter(self.loader_a)
        iter_b = iter(self.loader_b)

        while True:
            # Randomly choose which loader to sample from
            if torch.rand(1).item() < self.fraction_b:
                try:
                    batch = next(iter_b)
                    yield batch, "joint"
                except StopIteration:
                    iter_b = iter(self.loader_b)
                    batch = next(iter_b)
                    yield batch, "joint"
            else:
                try:
                    batch = next(iter_a)
                    yield batch, "gesture"
                except StopIteration:
                    iter_a = iter(self.loader_a)
                    batch = next(iter_a)
                    yield batch, "gesture"

    def __len__(self):
        return len(self.loader_a) + len(self.loader_b)
