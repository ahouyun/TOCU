#!/usr/bin/env python3
"""Fail-closed audit for a standalone formal Seattle TOCU integrated run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_INNOVATIONS = {
    "frequency_conditioned",
    "orthogonal_two_source",
    "frequency_impedance_variance_consistency",
}
# Keep the released protocol labels accepted and allow the isolated
# long-budget Seattle probe to be audited by the same fail-closed checks.
EXPECTED_PROTOCOLS = {
    "TOCU_integrated",
    "TOCU_integrated_rmse",
    "TOCU_integrated_rmse_tail_long",
    "TOCU_integrated_rmse_horizon_affine",
    "TOCU_integrated_rmse_horizon_affine_gated",
    "TOCU_integrated_rmse_horizon_node_affine",
    "TOCU_integrated_history_delta12_horizon10",
    "TOCU_integrated_history_graph_horizon10",
    "TOCU_integrated_history_graph_relative_horizon10",
    "TOCU_integrated_history_graph_relative_rmse_tail_affine",
    "TOCU_integrated_history_graph_relative_rmse_tail_conservative",
    "TOCU_integrated_history_graph_relative_rmse_tail_ultra_conservative",
    "TOCU_integrated_history_graph_bidirectional_horizon10",
}
SEEDS = (0, 1, 2)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def target_hash(target: np.ndarray) -> str:
    canonical = np.rint(target).astype(np.float32, copy=False)
    canonical[canonical == 0] = 0.0
    return hashlib.sha256(np.ascontiguousarray(canonical).tobytes()).hexdigest()


def recompute(summary_path: Path) -> tuple[dict[str, float], str, list[int]]:
    with np.load(summary_path) as summary:
        if set(summary.files) != {"mean", "lower", "upper", "target"}:
            raise RuntimeError(f"summary keys mismatch: {summary_path}")
        mean, lower, upper, target = (summary[key] for key in ("mean", "lower", "upper", "target"))
    if not (mean.shape == lower.shape == upper.shape == target.shape) or mean.ndim != 4:
        raise RuntimeError(f"summary shape mismatch: {summary_path}")
    if not all(np.isfinite(array).all() for array in (mean, lower, upper, target)):
        raise RuntimeError(f"non-finite summary: {summary_path}")
    if not np.all(lower <= upper):
        raise RuntimeError(f"invalid interval: {summary_path}")
    error = np.abs(mean - target)
    mask = target > 10.0
    interval = (upper - lower) + 40.0 * (
        np.maximum(lower - target, 0.0) + np.maximum(target - upper, 0.0)
    )
    metrics = {
        "MAE": float(error.mean(dtype=np.float64)),
        "RMSE": float(np.sqrt(np.mean((mean - target) ** 2, dtype=np.float64))),
        "MAPE": float(
            100.0 * np.sum(error[mask] / np.maximum(np.abs(target[mask]), 1e-5), dtype=np.float64)
            / max(int(mask.sum()), 1)
        ),
        "MIS": float(interval.mean(dtype=np.float64)),
    }
    return metrics, target_hash(target), list(target.shape)


def audit_seed(root: Path, seed: int) -> dict:
    seed_root = root / f"seed{seed}"
    required = (
        "DONE",
        "metrics.json",
        "history.json",
        "selection.json",
        "checkpoint.pt",
        "summary.npz",
    )
    missing = [name for name in required if not (seed_root / name).is_file() or (seed_root / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"seed{seed}: missing artifacts {missing}")
    metrics = read_json(seed_root / "metrics.json")
    if metrics.get("model") != "TOCU" or metrics.get("model_alias") != "TOCU":
        raise RuntimeError(f"seed{seed}: model identity mismatch")
    if metrics.get("dataset") != "Seattle" or int(metrics.get("seed", -1)) != seed or metrics.get("status") != "completed":
        raise RuntimeError(f"seed{seed}: provenance mismatch")
    if metrics.get("protocol_label") not in EXPECTED_PROTOCOLS:
        raise RuntimeError(f"seed{seed}: protocol mismatch")
    if metrics.get("training_mode") != "train_plus_validation_refit_TOCU_integrated":
        raise RuntimeError(f"seed{seed}: training mode mismatch")
    if tuple(metrics.get("split", ())) != (0.6, 0.2, 0.2) or metrics.get("input_len") != 12 or metrics.get("output_len") != 12:
        raise RuntimeError(f"seed{seed}: public configuration mismatch")
    if int(metrics.get("num_samples", -1)) != 50:
        raise RuntimeError(f"seed{seed}: sample count mismatch")
    if not EXPECTED_INNOVATIONS.issubset(set(metrics.get("innovation_flags", []))):
        raise RuntimeError(f"seed{seed}: innovation flags mismatch")
    if metrics.get("protocol_label") == "TOCU_integrated_rmse_horizon_affine":
        selected = metrics.get("ratio_calibration_selected") or {}
        if selected.get("family") != "horizon_affine":
            raise RuntimeError(f"seed{seed}: horizon-affine protocol did not select horizon_affine")
        if "validation_guarded_horizon_affine_correction" not in metrics.get("optimization_flags", []):
            raise RuntimeError(f"seed{seed}: horizon-affine optimization flag missing")
    if metrics.get("protocol_label") == "TOCU_integrated_rmse_horizon_affine_gated":
        selected = metrics.get("ratio_calibration_selected") or {}
        if selected.get("family") != "horizon_affine" or selected.get("gate") not in {"below", "above"}:
            raise RuntimeError(f"seed{seed}: gated horizon-affine selection missing")
        if "validation_guarded_horizon_affine_correction" not in metrics.get("optimization_flags", []):
            raise RuntimeError(f"seed{seed}: gated horizon-affine optimization flag missing")
    if metrics.get("protocol_label") == "TOCU_integrated_rmse_horizon_node_affine":
        selected = metrics.get("ratio_calibration_selected") or {}
        if selected.get("family") != "horizon_node_affine":
            raise RuntimeError(f"seed{seed}: horizon-node-affine protocol did not select horizon_node_affine")
        if "validation_guarded_horizon_node_affine_correction" not in metrics.get("optimization_flags", []):
            raise RuntimeError(f"seed{seed}: horizon-node-affine optimization flag missing")
    if metrics.get("protocol_label") == "TOCU_integrated_history_graph_relative_rmse_tail_conservative":
        selected = metrics.get("ratio_calibration_selected") or {}
        if selected.get("family") != "horizon_affine":
            raise RuntimeError(f"seed{seed}: conservative tail protocol did not select horizon_affine")
        if "validation_guarded_rmse_tail_affine_conservative_correction" not in metrics.get("optimization_flags", []):
            raise RuntimeError(f"seed{seed}: conservative tail optimization flag missing")
    if metrics.get("protocol_label") == "TOCU_integrated_history_graph_relative_rmse_tail_ultra_conservative":
        selected = metrics.get("ratio_calibration_selected") or {}
        if selected.get("family") != "horizon_affine":
            raise RuntimeError(f"seed{seed}: ultra-conservative tail protocol did not select horizon_affine")
        if "validation_guarded_rmse_tail_affine_ultra_conservative_correction" not in metrics.get("optimization_flags", []):
            raise RuntimeError(f"seed{seed}: ultra-conservative tail optimization flag missing")
    if metrics.get("protocol_label") in {
        "TOCU_integrated_history_graph_horizon10",
        "TOCU_integrated_history_graph_relative_horizon10",
        "TOCU_integrated_history_graph_relative_rmse_tail_affine",
        "TOCU_integrated_history_graph_relative_rmse_tail_conservative",
        "TOCU_integrated_history_graph_relative_rmse_tail_ultra_conservative",
        "TOCU_integrated_history_graph_bidirectional_horizon10",
    }:
        if metrics.get("history_graph_enabled") is not True:
            raise RuntimeError(f"seed{seed}: graph-history feature flag missing")
        graph_mode = metrics.get("history_graph_mode", "outgoing")
        if graph_mode not in {"outgoing", "bidirectional", "symmetric"}:
            raise RuntimeError(f"seed{seed}: unsupported graph-history mode {graph_mode!r}")
        if "causal_graph_history_adapter" not in metrics.get("optimization_flags", []):
            raise RuntimeError(f"seed{seed}: graph-history optimization flag missing")
        if graph_mode == "bidirectional" and "causal_graph_history_bidirectional" not in metrics.get("optimization_flags", []):
            raise RuntimeError(f"seed{seed}: bidirectional graph-history optimization flag missing")
        selection = read_json(seed_root / "selection.json")
        if selection.get("history_graph_enabled") is not True:
            raise RuntimeError(f"seed{seed}: selection graph-history flag missing")
        if selection.get("history_graph_mode", "outgoing") != graph_mode:
            raise RuntimeError(f"seed{seed}: graph-history mode provenance mismatch")
    selection = read_json(seed_root / "selection.json")
    if selection.get("dataset") != "Seattle" or int(selection.get("seed", -1)) != seed:
        raise RuntimeError(f"seed{seed}: selection provenance mismatch")
    if "test_read" not in str(selection.get("protocol", "")):
        raise RuntimeError(f"seed{seed}: selection test-read guard missing")
    history = read_json(seed_root / "history.json")
    for key in ("backbone", "adapter", "history_selection", "head"):
        if not history.get(key):
            raise RuntimeError(f"seed{seed}: incomplete history {key}")
    recomputed, digest, shape = recompute(seed_root / "summary.npz")
    if digest != metrics.get("target_sha256") or shape != metrics.get("target_shape"):
        raise RuntimeError(f"seed{seed}: target provenance mismatch")
    deltas = {key: abs(recomputed[key] - float(metrics[key])) for key in recomputed}
    limits = {"MAE": 5e-4, "RMSE": 5e-4, "MAPE": 5e-4, "MIS": 5e-3}
    if any(deltas[key] > limits[key] for key in deltas):
        raise RuntimeError(f"seed{seed}: metric recomputation mismatch {deltas}")
    return {
        "seed": seed,
        "metrics": {key: float(metrics[key]) for key in ("MAE", "RMSE", "MAPE", "CRPS", "MIS")},
        "target_sha256": digest,
        "target_shape": shape,
        "recomputed_deltas": deltas,
        "innovation_flags": sorted(EXPECTED_INNOVATIONS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-history-cutoff-mode",
        choices=("source", "train_point_max", "none"),
        default=None,
        help="Optionally require the formal run to record a specific train-only cutoff mode.",
    )
    args = parser.parse_args()
    rows = []
    for seed in SEEDS:
        row = audit_seed(args.root, seed)
        metrics = read_json(args.root / f"seed{seed}" / "metrics.json")
        selection = read_json(args.root / f"seed{seed}" / "selection.json")
        if args.require_history_cutoff_mode is not None:
            expected = args.require_history_cutoff_mode
            if metrics.get("history_cutoff_mode") != expected:
                raise RuntimeError(
                    f"seed{seed}: history_cutoff_mode mismatch: "
                    f"{metrics.get('history_cutoff_mode')} != {expected}"
                )
            if selection.get("history_cutoff_mode") != expected:
                raise RuntimeError(
                    f"seed{seed}: selection history_cutoff_mode mismatch: "
                    f"{selection.get('history_cutoff_mode')} != {expected}"
                )
            cutoff = metrics.get("history_point_cutoff")
            support_cutoff = selection.get("history_support_cutoff")
            if cutoff is None or support_cutoff is None or abs(float(cutoff) - float(support_cutoff)) > 1e-5:
                raise RuntimeError(f"seed{seed}: cutoff provenance mismatch")
            if expected == "train_point_max":
                if "history_cutoff_train_point_max" not in metrics.get("optimization_flags", []):
                    raise RuntimeError(f"seed{seed}: train-point cutoff optimization flag missing")
            elif "history_cutoff_train_point_max" in metrics.get("optimization_flags", []):
                raise RuntimeError(f"seed{seed}: source cutoff unexpectedly carries train-point cutoff flag")
        rows.append(row)
    hashes = {row["target_sha256"] for row in rows}
    if len(hashes) != 1:
        raise RuntimeError(f"seed target hashes differ: {hashes}")
    payload = {
        "status": "passed",
        "model": "TOCU",
        "dataset": "Seattle",
        "protocol": "TOCU_integrated_rmse_formal_fail_closed",
        "seeds": rows,
        "mean": {
            key: float(np.mean([row["metrics"][key] for row in rows]))
            for key in ("MAE", "RMSE", "MAPE", "CRPS", "MIS")
        },
        "sample_std": {
            key: float(np.std([row["metrics"][key] for row in rows], ddof=1))
            for key in ("MAE", "RMSE", "MAPE", "CRPS", "MIS")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
