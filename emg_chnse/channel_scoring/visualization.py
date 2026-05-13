"""
Visualization for channel scoring results.

Generates:
  1. Per-channel ranking bar chart
  2. Per-gesture channel activation heatmap
  3. Cross-channel correlation matrix
  4. Method agreement matrix
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend

# Chinese font support
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_channel_ranking(
    ranking: list[dict],
    top4: list[int],
    bottom4: list[int],
    title: str,
    save_path: str | Path,
):
    """Bar chart of channel rank-sum scores (lower = better)."""
    channels = [r["channel"] for r in ranking]
    rank_sums = [r["rank_sum"] for r in ranking]

    colors = []
    for ch in channels:
        if ch in top4:
            colors.append("#2ecc71")  # green
        elif ch in bottom4:
            colors.append("#e74c3c")  # red
        else:
            colors.append("#3498db")  # blue

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(16), rank_sums, color=colors, edgecolor="white")

    ax.set_xticks(range(16))
    ax.set_xticklabels([f"Ch {ch}" for ch in channels])
    ax.set_xlabel("Channel")
    ax.set_ylabel("Rank Sum (lower = better)")
    ax.set_title(title)
    ax.axhline(y=np.mean(rank_sums), color="gray", linestyle="--", alpha=0.5, label="Mean")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Top-4"),
        Patch(facecolor="#e74c3c", label="Bottom-4"),
        Patch(facecolor="#3498db", label="Middle"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_gesture_channel_heatmap(
    event_data: dict,
    scenario: str,
    save_path: str | Path,
):
    """
    Heatmap: (N_gestures × 16) showing mean activation per channel per gesture.
    """
    from .data_utils import SCENARIO_GESTURES

    gesture_names = SCENARIO_GESTURES[scenario]
    activation = np.zeros((len(gesture_names), 16))

    for i, name in enumerate(gesture_names):
        segs = event_data["signal"].get(name)
        if segs is not None and segs.ndim >= 3 and segs.shape[0] > 0:
            # RMS per channel
            rms = np.sqrt(np.mean(segs**2, axis=(0, 2)))  # (16,)
            activation[i] = rms

    # Normalize per gesture (row-wise)
    row_max = activation.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    activation_norm = activation / row_max

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(activation_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(16))
    ax.set_xticklabels([f"Ch {c}" for c in range(16)])
    ax.set_yticks(range(len(gesture_names)))
    ax.set_yticklabels(gesture_names)
    ax.set_xlabel("Channel")
    ax.set_title(f"Per-Gesture Channel Activation ({scenario})")

    plt.colorbar(im, ax=ax, label="Normalized RMS")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_correlation_matrix(
    corr_matrix: np.ndarray,
    top4: list[int],
    save_path: str | Path,
):
    """16×16 cross-channel correlation heatmap."""
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

    # Highlight top-4
    for ch in top4:
        ax.add_patch(plt.Rectangle((ch - 0.5, ch - 0.5), 1, 1,
                                    fill=False, edgecolor="#2ecc71", linewidth=2.5))

    ax.set_xticks(range(16))
    ax.set_xticklabels([f"Ch {c}" for c in range(16)], rotation=45)
    ax.set_yticks(range(16))
    ax.set_yticklabels([f"Ch {c}" for c in range(16)])
    ax.set_title("Cross-Channel Correlation Matrix\n(green box = Top-4)")

    plt.colorbar(im, ax=ax, label="Pearson r")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_method_agreement(
    per_method_ranks: dict[str, list[int]],
    save_path: str | Path,
):
    """
    Spearman correlation between method rankings.
    Shows how much methods agree with each other.
    """
    from scipy.stats import spearmanr

    method_names = list(per_method_ranks.keys())
    n = len(method_names)
    agreement = np.zeros((n, n))

    for i, m1 in enumerate(method_names):
        for j, m2 in enumerate(method_names):
            r, _ = spearmanr(per_method_ranks[m1], per_method_ranks[m2])
            agreement[i, j] = r

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(agreement, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels(method_names, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(method_names)
    ax.set_title("Method Agreement (Spearman ρ)")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{agreement[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if abs(agreement[i, j]) < 0.6 else "white")

    plt.colorbar(im, ax=ax, label="Spearman ρ")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_per_method_scores(
    per_method_scores: dict[str, np.ndarray],
    top4: list[int],
    scenario: str,
    save_path: str | Path,
):
    """
    Grid of bar charts — one per method, showing per-channel scores.
    """
    method_names = list(per_method_scores.keys())
    n = len(method_names)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(method_names):
        ax = axes[i]
        scores = per_method_scores[name]
        colors = ["#2ecc71" if c in top4 else "#3498db" for c in range(16)]
        ax.bar(range(16), scores, color=colors, edgecolor="white")
        ax.set_xticks(range(16))
        ax.set_xticklabels([f"{c}" for c in range(16)], fontsize=7)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Channel")

    # Hide unused subplots
    for j in range(len(method_names), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Per-Method Channel Scores — {scenario}", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")
