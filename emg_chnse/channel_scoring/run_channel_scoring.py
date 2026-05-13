#!/usr/bin/env python3
"""
Channel scoring pipeline orchestrator.

Runs the full pipeline:
  Phase 1 — Collect event-aligned EMG data from training HDF5 files
  Phase 2 — Run scoring methods (SNR, Fisher, MI, Weight Norm)
  Phase 3 — Aggregate and rank channels
  Phase 4 — Visualize results

Usage:
  python -m emg_chnse.channel_scoring.run_channel_scoring

Or with overrides:
  python -m emg_chnse.channel_scoring.run_channel_scoring \
      --scenario thumb \
      --data-location D:/emg/emg_nature/emg_data \
      --csv-path D:/emg/emg_nature/emg_data/discrete_gestures_corpus.csv \
      --ckpt-path D:/emg/emg2pose1/emg2pose_model_checkpoints/tracking_vemg2pose.ckpt \
      --output-dir D:/emg/emg_chnse/outputs \
      --max-files 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .data_utils import (
    NUM_CHANNELS,
    collect_scenario_event_data,
)
from .methods import (
    SNRScoring,
    FisherScoring,
    MutualInfoScoring,
    WeightNormScoring,
    SaliencyScoring,
    AblationScoring,
)
from .aggregate import (
    aggregate_scores,
    compute_correlation_matrix,
    check_redundancy,
    save_results,
)
from .visualization import (
    plot_channel_ranking,
    plot_gesture_channel_heatmap,
    plot_correlation_matrix,
    plot_method_agreement,
    plot_per_method_scores,
)


def parse_args():
    p = argparse.ArgumentParser(description="EMG Channel Scoring Pipeline")
    p.add_argument("--scenario", type=str, default="thumb",
                   choices=["thumb", "index_middle", "both"],
                   help="Gesture scenario to score")
    p.add_argument("--data-location", type=str,
                   default="D:/emg/emg_nature/emg_data",
                   help="Root directory of HDF5 files")
    p.add_argument("--csv-path", type=str,
                   default="D:/emg/emg_nature/emg_data/discrete_gestures_corpus.csv",
                   help="Path to data split CSV")
    p.add_argument("--ckpt-path", type=str,
                   default="D:/emg/emg2pose1/emg2pose_model_checkpoints/tracking_vemg2pose.ckpt",
                   help="Path to emg2pose pre-trained checkpoint (for weight-norm method)")
    p.add_argument("--output-dir", type=str,
                   default="D:/emg/emg_chnse/outputs",
                   help="Output directory for results and figures")
    p.add_argument("--cache-dir", type=str,
                   default="D:/emg/emg_chnse/outputs/cache",
                   help="Cache directory for extracted event data")
    p.add_argument("--max-files", type=int, default=None,
                   help="Max training HDF5 files to process (None = all ~80)")
    p.add_argument("--skip-saliency", action="store_true",
                   help="Skip saliency method (needs trained model + GPU)")
    p.add_argument("--skip-ablation", action="store_true",
                   help="Skip ablation method (needs trained model + GPU)")
    p.add_argument("--trained-model-ckpt", type=str, default=None,
                   help="Path to trained 16-channel model checkpoint (for saliency/ablation)")
    return p.parse_args()


def main():
    args = parse_args()

    scenarios = ["thumb", "index_middle"] if args.scenario == "both" else [args.scenario]

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    figures_dir = output_dir / "figures"
    for d in [output_dir, cache_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        print(f"WARNING: Checkpoint not found at {ckpt_path}. Weight-norm method will fail.")
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"ERROR: CSV split file not found at {csv_path}")
        sys.exit(1)

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"  Scenario: {scenario}")
        print(f"{'='*60}")

        # =====================================================================
        # Phase 1: Collect event-aligned EMG data
        # =====================================================================
        cache_file = cache_dir / f"{scenario}_event_data.pkl"
        print(f"\n[Phase 1] Collecting event-aligned EMG data...")
        event_data = collect_scenario_event_data(
            data_location=args.data_location,
            csv_path=csv_path,
            scenario=scenario,
            signal_before_ms=200.0,
            baseline_start_ms=500.0,
            baseline_end_ms=300.0,
            cache_path=cache_file,
            split="train",
            max_files=args.max_files,
        )
        # Print stats
        for name, segs in event_data["signal"].items():
            n = segs.shape[0] if segs.ndim >= 3 else 0
            print(f"  {name}: {n} events")

        # =====================================================================
        # Phase 2: Run scoring methods
        # =====================================================================
        print(f"\n[Phase 2] Running scoring methods...")
        per_method_scores = {}
        higher_is_better = {}

        # Method 1: SNR
        print("  [1/6] SNR...")
        snr = SNRScoring()
        scores_snr = snr.compute(event_data, scenario)
        per_method_scores["snr"] = scores_snr
        higher_is_better["snr"] = True
        print(f"    Top-3: {_top_channels(scores_snr, 3)}")

        # Method 2: Fisher
        print("  [2/6] Fisher...")
        fisher = FisherScoring()
        scores_fisher = fisher.compute(event_data, scenario)
        per_method_scores["fisher"] = scores_fisher
        higher_is_better["fisher"] = True
        print(f"    Top-3: {_top_channels(scores_fisher, 3)}")

        # Method 3: Mutual Information
        print("  [3/6] Mutual Information...")
        mi = MutualInfoScoring()
        scores_mi = mi.compute(event_data, scenario)
        per_method_scores["mutual_info"] = scores_mi
        higher_is_better["mutual_info"] = True
        print(f"    Top-3: {_top_channels(scores_mi, 3)}")

        # Method 4: Weight Norm (scenario-independent, cached internally)
        print("  [4/6] Weight Norm...")
        if ckpt_path.exists():
            wn = WeightNormScoring(checkpoint_path=ckpt_path)
            scores_wn = wn.compute(event_data, scenario)
            per_method_scores["weight_norm"] = scores_wn
            higher_is_better["weight_norm"] = True
            print(f"    Top-3: {_top_channels(scores_wn, 3)}")
        else:
            print("    SKIPPED (checkpoint not found)")

        # Method 5: Saliency (requires trained model)
        if not args.skip_saliency and args.trained_model_ckpt:
            print("  [5/6] Saliency...")
            scores_sal = _run_saliency(args, scenario)
            if scores_sal is not None:
                per_method_scores["saliency"] = scores_sal
                higher_is_better["saliency"] = True
                print(f"    Top-3: {_top_channels(scores_sal, 3)}")
        else:
            print("  [5/6] Saliency SKIPPED")

        # Method 6: Ablation (requires trained model)
        if not args.skip_ablation and args.trained_model_ckpt:
            print("  [6/6] Ablation...")
            scores_abl = _run_ablation(args, scenario)
            if scores_abl is not None:
                per_method_scores["ablation"] = scores_abl
                higher_is_better["ablation"] = True
                print(f"    Top-3: {_top_channels(scores_abl, 3)}")
        else:
            print("  [6/6] Ablation SKIPPED")

        # =====================================================================
        # Phase 3: Aggregate
        # =====================================================================
        print(f"\n[Phase 3] Aggregating scores...")
        results = aggregate_scores(per_method_scores, higher_is_better)

        # Correlation matrix
        print("  Computing cross-channel correlation...")
        corr = compute_correlation_matrix(event_data, scenario)

        # Redundancy check
        top4 = results["top4"]
        adjusted_top4, redundant = check_redundancy(top4, corr, results["ranking"])
        if redundant:
            print(f"  Redundancy detected: {redundant}")
            print(f"  Adjusted top-4: {adjusted_top4}")
        results["top4_adjusted"] = adjusted_top4
        results["redundant_pairs"] = redundant
        results["correlation_matrix"] = corr.tolist()

        # Print ranking
        print(f"\n  Final ranking (rank-sum, lower=better):")
        for i, entry in enumerate(results["ranking"]):
            marker = " ★" if entry["channel"] in adjusted_top4 else ""
            print(f"    {i+1:2d}. Ch {entry['channel']:2d}  "
                  f"rank_sum={entry['rank_sum']:3d}  z_mean={entry['z_mean']:+.3f}{marker}")

        print(f"\n  Recommended 4-channel subset: {adjusted_top4}")

        # Save
        save_results(results, output_dir, scenario)

        # =====================================================================
        # Phase 4: Visualize
        # =====================================================================
        print(f"\n[Phase 4] Generating figures...")

        plot_channel_ranking(
            results["ranking"], adjusted_top4, results["bottom4"],
            title=f"Channel Ranking — {scenario}",
            save_path=figures_dir / f"{scenario}_channel_ranking.png",
        )

        plot_gesture_channel_heatmap(
            event_data, scenario,
            save_path=figures_dir / f"{scenario}_gesture_heatmap.png",
        )

        plot_correlation_matrix(
            corr, adjusted_top4,
            save_path=figures_dir / f"{scenario}_correlation_matrix.png",
        )

        if len(per_method_scores) >= 2:
            plot_method_agreement(
                {n: results["per_method_ranks"][n] for n in per_method_scores},
                save_path=figures_dir / f"{scenario}_method_agreement.png",
            )

        plot_per_method_scores(
            per_method_scores, adjusted_top4, scenario,
            save_path=figures_dir / f"{scenario}_per_method_scores.png",
        )

    print(f"\n{'='*60}")
    print(f"  Done! Results saved to {output_dir}")
    print(f"{'='*60}")


def _top_channels(scores: np.ndarray, k: int = 3) -> list[int]:
    """Return indices of top-k channels (higher score = better)."""
    return list(np.argsort(scores)[::-1][:k])


def _run_saliency(args, scenario: str) -> np.ndarray | None:
    """Attempt to run saliency method. Returns None if dependencies missing."""
    try:
        from emg_transfer.lightning import DiscreteGesturesModule
        from emg_transfer.data_module import WindowedEmgDataModule
        from emg_transfer.data import DataSplit, make_dataset
        from emg_transfer.transforms import DiscreteGesturesTransform
        import torch
        from torch.utils.data import DataLoader
    except ImportError as e:
        print(f"    Cannot import emg_transfer: {e}")
        return None

    try:
        # Load the trained model
        ckpt = torch.load(args.trained_model_ckpt, map_location="cpu", weights_only=True)
        # Initialize the module from checkpoint hyperparameters
        hparams = ckpt.get("hyper_parameters", {})
        module = DiscreteGesturesModule.load_from_checkpoint(
            args.trained_model_ckpt, map_location="cpu"
        )
        model = module.network
        model.eval()

        # Build a small validation dataloader
        transform = DiscreteGesturesTransform(pulse_window=[0.08, 0.12])
        split = DataSplit.from_csv(args.csv_path)
        val_dataset = make_dataset(
            data_location=args.data_location,
            partition_dict=split.val,
            transform=transform,
            emg_augmentation=None,
            window_length=16000,
            stride=16000,
            jitter=False,
            split_label="saliency_val",
        )
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

        saliency = SaliencyScoring(
            model=model,
            dataloader=val_loader,
            num_batches=min(200, len(val_loader)),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        return saliency.compute(event_data=None, scenario=scenario)
    except Exception as e:
        print(f"    Saliency failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _run_ablation(args, scenario: str) -> np.ndarray | None:
    """Attempt to run ablation method. Returns None if dependencies missing."""
    try:
        from emg_transfer.lightning import DiscreteGesturesModule
        from emg_transfer.data import DataSplit, make_dataset
        from emg_transfer.transforms import DiscreteGesturesTransform
        import torch
    except ImportError as e:
        print(f"    Cannot import emg_transfer: {e}")
        return None

    try:
        module = DiscreteGesturesModule.load_from_checkpoint(
            args.trained_model_ckpt, map_location="cpu"
        )
        model = module.network
        model.eval()

        # Collect test data (full partitions, no windowing)
        transform = DiscreteGesturesTransform(pulse_window=[0.08, 0.12])
        split = DataSplit.from_csv(args.csv_path, pool_test_partitions=True)
        test_dataset = make_dataset(
            data_location=args.data_location,
            partition_dict=split.test,
            transform=transform,
            emg_augmentation=None,
            window_length=None,
            stride=None,
            jitter=False,
            split_label="ablation_test",
        )

        # For each test sample (full recording), collect data
        test_samples = []
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            test_samples.append(sample)

        ablation = AblationScoring(
            model=model,
            test_data=test_samples,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        return ablation.compute(event_data=None, scenario=scenario)
    except Exception as e:
        print(f"    Ablation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
