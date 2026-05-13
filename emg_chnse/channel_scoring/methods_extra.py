"""
Extended channel scoring methods for robustness and complementarity.

Method 7: Per-Channel Logistic Regression — simple but interpretable
Method 8: Gesture-Pair Activation Difference — physiological contrast
Method 9: Greedy Forward Selection — captures channel complementarity
Method 10: Bootstrap Stability — measures ranking confidence
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .data_utils import NUM_CHANNELS, SCENARIO_GESTURES


# =============================================================================
# Method 7: Per-Channel Logistic Regression
# =============================================================================

def score_logistic_regression(
    event_data: dict,
    scenario: str,
    window_ms: tuple = (-200.0, 0.0),
    n_folds: int = 5,
) -> np.ndarray:
    """
    For each channel independently, train a logistic regression classifier
    using the mean EMG value in the event window as the feature.

    Score = mean cross-validated accuracy across folds.

    This directly measures: "if I only had this one channel, how well could I
    distinguish the gestures in this scenario?"

    Returns:
        np.ndarray(16,) — per-channel accuracy scores
    """
    gesture_names = SCENARIO_GESTURES[scenario]

    # Collect features: per-event mean EMG
    X_all = []  # (N_events, 16)
    y_all = []  # (N_events,)
    for g_idx, name in enumerate(gesture_names):
        segs = event_data["signal"].get(name)
        if segs is None or segs.ndim < 3 or segs.shape[0] == 0:
            continue
        feat = np.mean(segs, axis=-1)  # (N, 16)
        X_all.append(feat)
        y_all.append(np.full(feat.shape[0], g_idx))

    if len(X_all) < 2:
        return np.zeros(NUM_CHANNELS)

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    scores = np.zeros(NUM_CHANNELS)
    for c in range(NUM_CHANNELS):
        X_c = X[:, c:c+1]  # (N, 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_c)

        # Use balanced class weights to handle class imbalance
        clf = LogisticRegression(
            max_iter=500,
            class_weight='balanced',
            random_state=42,
        )
        try:
            cv_scores = cross_val_score(clf, X_scaled, y, cv=min(n_folds, 3), scoring='accuracy')
            scores[c] = np.mean(cv_scores)
        except Exception:
            scores[c] = 0.0

    return scores


# =============================================================================
# Method 8: Gesture-Pair Activation Difference
# =============================================================================

def score_gesture_pair_diff(
    event_data: dict,
    scenario: str,
) -> np.ndarray:
    """
    For each pair of gestures in the scenario, compute the difference in
    per-channel RMS activation. Sum the absolute differences across all pairs.

    This measures physiological contrast: channels that show DIFFERENT
    activation levels for different gestures are more informative.

    Score(c) = Σ_{g1,g2} |RMS(c|g1) - RMS(c|g2)|

    Returns:
        np.ndarray(16,) — per-channel contrast scores
    """
    gesture_names = SCENARIO_GESTURES[scenario]

    # Compute per-gesture per-channel RMS
    rms_per_gesture = {}
    for name in gesture_names:
        segs = event_data["signal"].get(name)
        if segs is None or segs.ndim < 3 or segs.shape[0] == 0:
            continue
        rms = np.sqrt(np.mean(segs**2, axis=(0, 2)))  # (16,)
        rms_per_gesture[name] = rms

    if len(rms_per_gesture) < 2:
        return np.zeros(NUM_CHANNELS)

    # Sum pairwise absolute differences
    scores = np.zeros(NUM_CHANNELS)
    gesture_list = list(rms_per_gesture.keys())
    n_pairs = 0
    for i in range(len(gesture_list)):
        for j in range(i + 1, len(gesture_list)):
            scores += np.abs(
                rms_per_gesture[gesture_list[i]] -
                rms_per_gesture[gesture_list[j]]
            )
            n_pairs += 1

    if n_pairs > 0:
        scores /= n_pairs

    return scores


# =============================================================================
# Method 9: Greedy Forward Selection
# =============================================================================

def score_greedy_forward_selection(
    event_data: dict,
    scenario: str,
    max_channels: int = 8,
    n_folds: int = 5,
) -> tuple[np.ndarray, list[int]]:
    """
    Greedy forward selection considering channel complementarity.

    Starts with no channels, iteratively adds the channel that most improves
    cross-validated logistic regression accuracy.

    This captures INTER-channel complementarity, not just individual quality.

    Returns:
        (scores, selected_order)
        scores: (16,) — 0 for never-selected, rank score for selected
        selected_order: list of channel indices in selection order
    """
    gesture_names = SCENARIO_GESTURES[scenario]

    # Build feature matrix: per-event mean EMG, all 16 channels
    X_list, y_list = [], []
    for g_idx, name in enumerate(gesture_names):
        segs = event_data["signal"].get(name)
        if segs is None or segs.ndim < 3 or segs.shape[0] == 0:
            continue
        feat = np.mean(segs, axis=-1)  # (N, 16)
        X_list.append(feat)
        y_list.append(np.full(feat.shape[0], g_idx))

    if len(X_list) < 2:
        return np.zeros(NUM_CHANNELS), []

    X_full = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X_full)

    available = list(range(NUM_CHANNELS))
    selected = []
    best_scores = []

    for _ in range(min(max_channels, NUM_CHANNELS)):
        best_ch = -1
        best_acc = -1.0
        for c in tqdm(available, desc=f"  Greedy step {len(selected)+1}", leave=False):
            cols = selected + [c]
            X_sub = X_full[:, cols]
            clf = LogisticRegression(
                max_iter=500, class_weight='balanced', random_state=42
            )
            try:
                cv_scores = cross_val_score(clf, X_sub, y, cv=min(n_folds, 3), scoring='accuracy')
                acc = np.mean(cv_scores)
            except Exception:
                acc = 0.0
            if acc > best_acc:
                best_acc = acc
                best_ch = c

        if best_ch >= 0:
            selected.append(best_ch)
            available.remove(best_ch)
            best_scores.append(best_acc)

    # Convert to per-channel scores: earlier selection = higher score
    scores = np.zeros(NUM_CHANNELS)
    for rank, ch in enumerate(selected):
        scores[ch] = float(max_channels - rank)  # 8, 7, 6, ...

    return scores, selected


# =============================================================================
# Method 10: Bootstrap Stability
# =============================================================================

def bootstrap_stability_analysis(
    event_data: dict,
    scenario: str,
    n_bootstraps: int = 20,
    sample_frac: float = 0.8,
    random_seed: int = 42,
) -> dict:
    """
    Run the scoring pipeline on multiple random subsets of the data to
    measure ranking stability.

    For each channel, computes:
      - mean_rank: average rank across bootstraps
      - std_rank: standard deviation of rank
      - top4_freq: fraction of bootstraps where channel appeared in top-4
      - top6_freq: fraction of bootstraps where channel appeared in top-6

    Returns:
        dict with stability metrics per channel
    """
    from .methods import SNRScoring, FisherScoring, MutualInfoScoring

    gesture_names = SCENARIO_GESTURES[scenario]
    rng = np.random.RandomState(random_seed)

    # Build full event lists per gesture
    all_events = {}
    for name in gesture_names:
        segs = event_data["signal"].get(name)
        if segs is not None and segs.ndim >= 3:
            all_events[name] = segs
        else:
            all_events[name] = np.array([])

    all_rankings = []
    methods_list = [SNRScoring(), FisherScoring(), MutualInfoScoring()]

    for boot_idx in tqdm(range(n_bootstraps), desc="  Bootstrap"):
        # Sample events per gesture
        boot_data = {"signal": {}, "baseline": {}}
        for name in gesture_names:
            segs = all_events[name]
            if segs.shape[0] == 0:
                boot_data["signal"][name] = np.array([])
                boot_data["baseline"][name] = np.array([])
                continue
            n_sample = max(2, int(segs.shape[0] * sample_frac))
            idxs = rng.choice(segs.shape[0], n_sample, replace=True)
            boot_data["signal"][name] = segs[idxs]
            # Use same indices for baseline
            base_segs = event_data["baseline"].get(name)
            if base_segs is not None and base_segs.ndim >= 3 and base_segs.shape[0] > 0:
                base_idxs = np.clip(idxs, 0, base_segs.shape[0] - 1)
                boot_data["baseline"][name] = base_segs[base_idxs]
            else:
                boot_data["baseline"][name] = np.array([])

        # Score this bootstrap
        rank_sum = np.zeros(NUM_CHANNELS)
        for method in methods_list:
            try:
                raw = method.compute(boot_data, scenario)
                # Rank (1=best)
                order = np.argsort(raw)[::-1]
                for rank, idx in enumerate(order):
                    rank_sum[idx] += rank + 1
                # Ensure baseline data available for SNR
                if isinstance(method, SNRScoring) and boot_data["baseline"].get(gesture_names[0], np.array([])).size == 0:
                    # If no baseline, give all same rank
                    pass
            except Exception:
                pass

        # Convert to ranking
        order = np.argsort(rank_sum)
        ranking = np.zeros(NUM_CHANNELS, dtype=int)
        for rank, idx in enumerate(order):
            ranking[idx] = rank + 1
        all_rankings.append(ranking)

    if not all_rankings:
        return {}

    all_rankings = np.array(all_rankings)  # (n_boot, 16)

    stability = {}
    for c in range(NUM_CHANNELS):
        ranks = all_rankings[:, c]
        stability[f"ch{c}"] = {
            "mean_rank": float(np.mean(ranks)),
            "std_rank": float(np.std(ranks)),
            "top4_freq": float(np.mean(ranks <= 4)),
            "top6_freq": float(np.mean(ranks <= 6)),
        }

    return stability


# =============================================================================
# Master: compute all extra scores
# =============================================================================

def compute_extra_scores(
    event_data: dict,
    scenario: str,
    n_bootstraps: int = 20,
) -> dict:
    """
    Compute all extra scoring methods and return consolidated results.

    Returns:
        dict with keys:
          - "logistic": (16,) per-channel accuracy
          - "pairwise_diff": (16,) gesture-pair contrast
          - "greedy_selection": list of selected channels in order
          - "greedy_scores": (16,) greedy selection scores
          - "bootstrap_stability": dict per channel
    """
    print(f"\n[Extra Methods] Scenario: {scenario}")

    # Method 7: Logistic Regression
    print("  [7] Per-channel Logistic Regression...")
    logistic_scores = score_logistic_regression(event_data, scenario)
    top3 = np.argsort(logistic_scores)[::-1][:3]
    print(f"    Top-3: {top3.tolist()} (acc: {[f'{logistic_scores[i]:.3f}' for i in top3]})")

    # Method 8: Gesture-Pair Difference
    print("  [8] Gesture-Pair Activation Difference...")
    pair_scores = score_gesture_pair_diff(event_data, scenario)
    top3 = np.argsort(pair_scores)[::-1][:3]
    print(f"    Top-3: {top3.tolist()}")

    # Method 9: Greedy Forward Selection
    print("  [9] Greedy Forward Selection...")
    greedy_scores, greedy_order = score_greedy_forward_selection(event_data, scenario)
    print(f"    Selection order: {greedy_order}")

    # Method 10: Bootstrap Stability
    print(f"  [10] Bootstrap Stability ({n_bootstraps} iterations)...")
    stability = bootstrap_stability_analysis(event_data, scenario, n_bootstraps=n_bootstraps)

    return {
        "logistic": logistic_scores.tolist(),
        "pairwise_diff": pair_scores.tolist(),
        "greedy_order": greedy_order,
        "greedy_scores": greedy_scores.tolist(),
        "bootstrap_stability": stability,
    }
