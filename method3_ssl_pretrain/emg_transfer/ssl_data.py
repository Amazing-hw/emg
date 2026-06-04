# Method 3: Data loading for self-supervised pre-training
#
# Loads raw EMG windows from emg2pose HDF5 files with no label requirements.

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader


class UnlabeledEmgDataset(Dataset):
    """
    Dataset returning unlabeled EMG windows from HDF5 files.

    Compatible with both emg2pose and emg_nature HDF5 formats —
    just reads the raw EMG signal without any labels.
    """

    def __init__(
        self,
        hdf5_paths: list[Path],
        window_length: int = 16000,
        stride: int = 8000,
        max_windows_per_file: int = 50,
    ):
        self.hdf5_paths = hdf5_paths
        self.window_length = window_length
        self.stride = stride
        self.max_windows_per_file = max_windows_per_file

        # Pre-compute valid windows for each file
        self.windows: list[tuple[int, int, int]] = []  # (file_idx, start_sample, end_sample)

        for i, path in enumerate(hdf5_paths):
            with h5py.File(path, 'r') as f:
                # Find the EMG data — handle different HDF5 structures
                # emg2pose format: f['emg2pose']['timeseries'] → shape (T,)
                # emg_nature format: f['data'] → shape (T,)
                total_samples = None
                if 'emg2pose' in f:
                    total_samples = f['emg2pose']['timeseries'].shape[0]
                elif 'data' in f:
                    total_samples = f['data'].shape[0]
                elif 'timeseries' in f:
                    total_samples = f['timeseries'].shape[0]

                if total_samples is None or total_samples < window_length:
                    continue

                num_windows = min(
                    (total_samples - window_length) // stride + 1,
                    max_windows_per_file,
                )
                for w in range(num_windows):
                    start = w * stride
                    end = start + window_length
                    self.windows.append((i, start, end))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        file_idx, start, end = self.windows[idx]
        path = self.hdf5_paths[file_idx]

        with h5py.File(path, 'r') as f:
            if 'emg2pose' in f:
                emg_data = f['emg2pose']['timeseries']['emg'][start:end]
            elif 'data' in f:
                emg_data = f['data']['emg'][start:end]
            elif 'timeseries' in f:
                emg_data = f['timeseries']['emg'][start:end]
            else:
                raise KeyError(f"Could not find EMG data in {path}")

        # Convert to tensor: (T, C) → (C, T)
        emg = torch.from_numpy(emg_data.astype(np.float32))
        emg = emg.permute(1, 0)  # (16, T)

        return {"emg": emg}


def create_ssl_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 8,
    window_length: int = 16000,
    stride: int = 8000,
    max_files: int = 200,
    max_windows_per_file: int = 50,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for SSL pre-training from a directory of HDF5 files.

    Args:
        data_dir: Directory containing HDF5 files
        split: 'train' or 'val' (for train/val split)
        batch_size: Batch size
        window_length: EMG window length in samples
        stride: Stride between windows
        max_files: Maximum number of files to use
        max_windows_per_file: Maximum windows per HDF5 file
        num_workers: DataLoader workers
    """
    data_path = Path(data_dir)
    hdf5_files = sorted(data_path.glob("*.hdf5"))

    if len(hdf5_files) == 0:
        data_path = Path(data_dir) / "emg2pose_data"
        hdf5_files = sorted(data_path.glob("*.hdf5"))

    if len(hdf5_files) == 0:
        # Try the emg2pose-specific path
        data_path = Path("D:/emg/emg2pose1/emg2pose-main/emg2pose-main/emg2pose_dataset/emg2pose_data")
        hdf5_files = sorted(data_path.glob("*.hdf5"))

    hdf5_files = hdf5_files[:max_files]

    # Train/val split (90/10)
    split_idx = int(len(hdf5_files) * 0.9)
    if split == "train":
        hdf5_files = hdf5_files[:split_idx]
    else:
        hdf5_files = hdf5_files[split_idx:]

    if len(hdf5_files) == 0:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}")

    dataset = UnlabeledEmgDataset(
        hdf5_paths=hdf5_files,
        window_length=window_length,
        stride=stride,
        max_windows_per_file=max_windows_per_file,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
