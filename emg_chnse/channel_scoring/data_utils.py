"""
Event-aligned EMG window extraction for channel scoring.

Reuses the HDF5 data format from emg_nature / emg_transfer.
Each HDF5 file contains:
  - data/emg  : (N, 16) float32  — 16-channel EMG at 2000 Hz
  - data/time : (N,)   float64  — absolute unix timestamps
  - prompts   : DataFrame with columns [name, time] — gesture event labels

Scenarios:
  A (thumb):       thumb_click, thumb_down, thumb_in, thumb_out, thumb_up
  B (index_middle): index_press, index_release, middle_press, middle_release
"""

from __future__ import annotations

import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ---- Constants ----
EMG_SAMPLE_RATE = 2000
NUM_CHANNELS = 16

# Scenario gesture name lists (matches GestureType enum in emg_transfer.constants)
SCENARIO_GESTURES = {
    "thumb": [
        "thumb_click",
        "thumb_down",
        "thumb_in",
        "thumb_out",
        "thumb_up",
    ],
    "index_middle": [
        "index_press",
        "index_release",
        "middle_press",
        "middle_release",
    ],
}

ALL_GESTURE_NAMES = [
    "index_press",
    "index_release",
    "middle_press",
    "middle_release",
    "thumb_click",
    "thumb_down",
    "thumb_in",
    "thumb_out",
    "thumb_up",
]

GESTURE_TO_IDX = {name: i for i, name in enumerate(ALL_GESTURE_NAMES)}


def get_scenario_gestures(scenario: str) -> list[str]:
    """Return the list of gesture names for a given scenario."""
    if scenario not in SCENARIO_GESTURES:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Choose from {list(SCENARIO_GESTURES.keys())}"
        )
    return SCENARIO_GESTURES[scenario]


def get_scenario_indices(scenario: str) -> list[int]:
    """Return the 0-based class indices for a given scenario."""
    return [GESTURE_TO_IDX[name] for name in SCENARIO_GESTURES[scenario]]


# ---- Event-aligned window extraction ----


def read_hdf5_prompts(hdf5_path: str | Path) -> pd.DataFrame:
    """Read the prompts DataFrame from an HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        if "prompts" not in f:
            raise KeyError(f"No 'prompts' group in {hdf5_path}")
        prompts = pd.read_hdf(str(hdf5_path), "prompts")
    return prompts


def read_hdf5_timestamps(hdf5_path: str | Path) -> np.ndarray:
    """Read the time column from an HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        timestamps = f["data"]["time"][:]
    return timestamps


def read_hdf5_emg(hdf5_path: str | Path) -> np.ndarray:
    """Read the full EMG array from an HDF5 file. Shape: (N, 16)."""
    with h5py.File(hdf5_path, "r") as f:
        emg = f["data"]["emg"][:]
    return emg


def extract_event_and_baseline_windows(
    hdf5_path: str | Path,
    gesture_names: list[str],
    signal_before_ms: float = 200.0,
    baseline_start_ms: float = 500.0,
    baseline_end_ms: float = 300.0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Extract both signal and baseline windows from a single HDF5 file.

    Signal window:  [t - signal_before_ms,  t]
    Baseline window: [t - baseline_start_ms, t - baseline_end_ms]

    Reads the EMG file once for efficiency.

    Returns:
        (signal_dict, baseline_dict)
        Each dict maps gesture_name → np.ndarray of shape (N_events, 16, N_time)
    """
    signal_before = int(signal_before_ms / 1000.0 * EMG_SAMPLE_RATE)
    base_start = int(baseline_start_ms / 1000.0 * EMG_SAMPLE_RATE)
    base_end = int(baseline_end_ms / 1000.0 * EMG_SAMPLE_RATE)

    timestamps = read_hdf5_timestamps(hdf5_path)
    prompts = read_hdf5_prompts(hdf5_path)
    emg = read_hdf5_emg(hdf5_path)

    signal_results: dict[str, list[np.ndarray]] = {name: [] for name in gesture_names}
    baseline_results: dict[str, list[np.ndarray]] = {name: [] for name in gesture_names}

    for _, row in prompts.iterrows():
        name = row["name"]
        if name not in gesture_names:
            continue

        t_event = row["time"]
        idx = np.searchsorted(timestamps, t_event)
        if idx <= base_start or idx >= len(timestamps):
            continue

        # Signal: [idx - signal_before, idx]
        sig_start = max(0, idx - signal_before)
        sig_end = idx
        if sig_end - sig_start >= signal_before // 2:
            seg = emg[sig_start:sig_end, :]  # (T_sig, 16)
            signal_results[name].append(seg.T)  # (16, T_sig)

        # Baseline: [idx - base_start, idx - base_end]
        bl_start = idx - base_start
        bl_end = idx - base_end
        if bl_end - bl_start >= (base_start - base_end) // 2:
            seg = emg[bl_start:bl_end, :]  # (T_base, 16)
            baseline_results[name].append(seg.T)  # (16, T_base)

    signal_out = {
        name: np.stack(segs, axis=0) if segs else np.array([])
        for name, segs in signal_results.items()
    }
    baseline_out = {
        name: np.stack(segs, axis=0) if segs else np.array([])
        for name, segs in baseline_results.items()
    }
    return signal_out, baseline_out




def load_split_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load the data split CSV (discrete_gestures_corpus.csv)."""
    return pd.read_csv(csv_path)


def get_hdf5_paths_for_split(
    data_location: str | Path,
    csv_path: str | Path,
    split: str = "train",
) -> list[Path]:
    """
    Get all HDF5 file paths for a given data split.

    Args:
        data_location: Root directory containing .hdf5 files.
        csv_path: Path to discrete_gestures_corpus.csv.
        split: One of "train", "val", "test".

    Returns:
        List of absolute HDF5 paths.
    """
    df = load_split_csv(csv_path)
    data_loc = Path(data_location)
    datasets = df[df["split"] == split]["dataset"].unique()
    paths = []
    for ds in datasets:
        p = data_loc / ds
        if not p.suffix:
            p = p.with_suffix(".hdf5")
        if p.exists():
            paths.append(p)
    return sorted(paths)


def collect_scenario_event_data(
    data_location: str | Path,
    csv_path: str | Path,
    scenario: str,
    signal_before_ms: float = 200.0,
    baseline_start_ms: float = 500.0,
    baseline_end_ms: float = 300.0,
    cache_path: str | Path | None = None,
    split: str = "train",
    max_files: int | None = None,
) -> dict:
    """
    Collect event-aligned EMG windows across all training files for a scenario.

    Signal window:  [t - signal_before_ms, t]
    Baseline window: [t - baseline_start_ms, t - baseline_end_ms]

    If cache_path is provided and the file exists, load from cache.
    Otherwise extract from HDF5 files and optionally save to cache.

    Returns:
        dict with keys:
          - "signal": {gesture_name: np.ndarray(N_events, 16, signal_time)}
          - "baseline": {gesture_name: np.ndarray(N_events, 16, baseline_time)}
          - "scenario": str
          - "gesture_names": list[str]
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        print(f"  Loading cached event data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    gesture_names = get_scenario_gestures(scenario)
    hdf5_paths = get_hdf5_paths_for_split(data_location, csv_path, split=split)

    if max_files is not None:
        hdf5_paths = hdf5_paths[:max_files]

    print(f"  Processing {len(hdf5_paths)} HDF5 files...")

    all_signal: dict[str, list[np.ndarray]] = {name: [] for name in gesture_names}
    all_baseline: dict[str, list[np.ndarray]] = {name: [] for name in gesture_names}

    for hdf5_path in tqdm(hdf5_paths, desc=f"[{scenario}] Extracting"):
        try:
            signal, baseline = extract_event_and_baseline_windows(
                hdf5_path, gesture_names,
                signal_before_ms=signal_before_ms,
                baseline_start_ms=baseline_start_ms,
                baseline_end_ms=baseline_end_ms,
            )
            for name in gesture_names:
                if signal[name].size > 0:
                    all_signal[name].append(signal[name])
                if baseline[name].size > 0:
                    all_baseline[name].append(baseline[name])
        except Exception as e:
            print(f"  Skipping {hdf5_path}: {e}")

    # Concatenate across files
    result = {
        "signal": {},
        "baseline": {},
        "scenario": scenario,
        "gesture_names": gesture_names,
    }
    for name in gesture_names:
        result["signal"][name] = (
            np.concatenate(all_signal[name], axis=0)
            if all_signal[name]
            else np.array([])
        )
        result["baseline"][name] = (
            np.concatenate(all_baseline[name], axis=0)
            if all_baseline[name]
            else np.array([])
        )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        print(f"  Cached to {cache_path}")

    return result
