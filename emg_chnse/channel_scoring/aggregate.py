"""
Score aggregation and channel subset recommendation.

Aggregates per-method scores into a final ranking using:
  1. Rank-sum (primary): rank channels 1-16 per method, sum ranks.
  2. Z-score mean (secondary): z-normalize each method, average z-scores.

Also handles redundancy checking via cross-channel correlation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_utils import NUM_CHANNELS


def _rank_scores(scores: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """
    Convert raw scores to ranks (1 = best, 16 = worst).

    Args:
        scores: (16,) raw scores.
        higher_is_better: if True, higher score → rank 1.
    """
    order = np.argsort(scores)
    if higher_is_better:
        order = order[::-1]
    ranks = np.zeros(NUM_CHANNELS, dtype=int)
    for rank, idx in enumerate(order):
        ranks[idx] = rank + 1
    return ranks


def _z_normalize(scores: np.ndarray) -> np.ndarray:
    """Z-normalize scores to zero mean, unit variance."""
    std = np.std(scores)
    if std < 1e-12:
        return np.zeros_like(scores)
    return (scores - np.mean(scores)) / std


def aggregate_scores(
    per_method_scores: dict[str, np.ndarray],
    higher_is_better: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """
    Aggregate per-method channel scores into final rankings.

    Args:
        per_method_scores: {"snr": (16,), "fisher": (16,), ...}
        higher_is_better: per-method flag. Defaults to True for all.

    Returns:
        dict with keys:
          - "per_method_raw": the input scores
          - "per_method_ranks": rank per method
          - "per_method_z": z-score per method
          - "rank_sum": (16,) lower = better
          - "z_mean": (16,) higher = better
          - "ranking": list of (channel_idx, rank_sum, z_mean) sorted best→worst
    """
    if higher_is_better is None:
        higher_is_better = {name: True for name in per_method_scores}

    method_names = list(per_method_scores.keys())
    n_methods = len(method_names)

    # Initialize arrays
    rank_matrix = np.zeros((n_methods, NUM_CHANNELS))
    z_matrix = np.zeros((n_methods, NUM_CHANNELS))

    for i, name in enumerate(method_names):
        raw = per_method_scores[name]
        hb = higher_is_better.get(name, True)
        rank_matrix[i] = _rank_scores(raw, higher_is_better=hb)
        z_matrix[i] = _z_normalize(raw)
        if not hb:
            z_matrix[i] = -z_matrix[i]  # flip so higher z = better

    rank_sum = rank_matrix.sum(axis=0)  # (16,), lower = better
    z_mean = z_matrix.mean(axis=0)  # (16,), higher = better

    # Ranking list
    order = np.argsort(rank_sum)  # best first
    ranking = [
        {
            "channel": int(idx),
            "rank_sum": int(rank_sum[idx]),
            "z_mean": float(z_mean[idx]),
        }
        for idx in order
    ]

    return {
        "per_method_raw": {n: scores.tolist() for n, scores in per_method_scores.items()},
        "per_method_ranks": {n: rank_matrix[i].tolist() for i, n in enumerate(method_names)},
        "per_method_z": {n: z_matrix[i].tolist() for i, n in enumerate(method_names)},
        "rank_sum": rank_sum.tolist(),
        "z_mean": z_mean.tolist(),
        "ranking": ranking,
        "top4": [int(idx) for idx in order[:4]],
        "bottom4": [int(idx) for idx in order[-4:]],
    }


def compute_correlation_matrix(event_data: dict, scenario: str) -> np.ndarray:
    """
    Compute pairwise Pearson correlation between channels using signal windows.

    Returns:
        (16, 16) correlation matrix.
    """
    from .data_utils import SCENARIO_GESTURES

    gesture_names = SCENARIO_GESTURES[scenario]
    all_segments = []
    for name in gesture_names:
        segs = event_data["signal"].get(name)
        if segs is not None and segs.ndim >= 3 and segs.shape[0] > 0:
            # segs: (N, 16, T) → flatten to (N*T, 16)
            all_segments.append(segs.transpose(0, 2, 1).reshape(-1, NUM_CHANNELS))

    if not all_segments:
        return np.eye(NUM_CHANNELS)

    X = np.concatenate(all_segments, axis=0)  # (N_total*T, 16)
    corr = np.corrcoef(X.T)  # (16, 16)
    return corr


def check_redundancy(
    top4: list[int],
    corr_matrix: np.ndarray,
    ranking: list[dict],
    threshold: float = 0.9,
) -> tuple[list[int], list[tuple[int, int, float]]]:
    """
    Check for highly correlated channels in the top-4.

    If two channels have |r| > threshold, replace the lower-ranked one
    with the next-best uncorrelated channel.

    Returns:
        (adjusted_top4, redundancy_pairs)
    """
    adjusted = list(top4)
    redundant_pairs = []

    for i in range(len(adjusted)):
        for j in range(i + 1, len(adjusted)):
            ci, cj = adjusted[i], adjusted[j]
            r = corr_matrix[ci, cj]
            if abs(r) > threshold:
                redundant_pairs.append((ci, cj, float(r)))

    if redundant_pairs:
        # Replace the lower-ranked channel
        for ci, cj, r in redundant_pairs:
            # Find which one ranks lower (higher rank_sum)
            rank_i = next(x["rank_sum"] for x in ranking if x["channel"] == ci)
            rank_j = next(x["rank_sum"] for x in ranking if x["channel"] == cj)
            to_replace = ci if rank_i > rank_j else cj

            # Find next best uncorrelated channel
            for entry in ranking:
                candidate = entry["channel"]
                if candidate in adjusted:
                    continue
                # Check correlation with all kept channels
                max_corr = max(abs(corr_matrix[candidate, kept]) for kept in adjusted if kept != to_replace)
                if max_corr < threshold:
                    adjusted[adjusted.index(to_replace)] = candidate
                    break

    return adjusted, redundant_pairs


def save_results(
    results: dict,
    output_dir: str | Path,
    scenario: str,
):
    """Save aggregation results to JSON and CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / f"{scenario}_aggregate.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")

    # CSV ranking table
    df = pd.DataFrame(results["ranking"])
    df.index = df.index + 1  # 1-based rank
    df.index.name = "rank"
    csv_path = output_dir / f"{scenario}_ranking.csv"
    df.to_csv(csv_path)
    print(f"Saved: {csv_path}")

    # CSV per-method scores
    scores_df = pd.DataFrame(results["per_method_raw"], index=[f"ch{c}" for c in range(NUM_CHANNELS)])
    scores_csv = output_dir / f"{scenario}_per_method_scores.csv"
    scores_df.to_csv(scores_csv)
    print(f"Saved: {scores_csv}")
