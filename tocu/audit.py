#!/usr/bin/env python3
"""Fail-closed acceptance audit for the formal TOCU integrated queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DATASETS = ("PEMS03", "PEMS04", "PEMS08", "Seattle")
SEEDS = (0, 1, 2)
METRICS = ("MAE", "RMSE", "MAPE", "CRPS", "MIS")
EXPECTED_INNOVATIONS = {
    "frequency_conditioned",
    "orthogonal_two_source",
    "frequency_impedance_variance_consistency",
}
EXPECTED_PROTOCOLS = {
    "TOCU_integrated",
    "TOCU_integrated_rmse",
    "TOCU_integrated_rmse_tail_long",
    "TOCU_integrated_history_graph_relative_rmse_tail_affine",
    "TOCU_integrated_history_graph_relative_rmse_tail_conservative",
    "TOCU_integrated_history_graph_relative_rmse_tail_ultra_conservative",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def recompute_summary(path: Path, dataset: str) -> tuple[dict[str, float], str, list[int]]:
    with np.load(path) as summary:
        required = {"mean", "lower", "upper", "target"}
        if set(summary.files) != required:
            raise RuntimeError(f"summary keys mismatch: {path}")
        mean, lower, upper, target = (summary[name] for name in ("mean", "lower", "upper", "target"))
        if not (mean.shape == lower.shape == upper.shape == target.shape) or mean.ndim != 4:
            raise RuntimeError(f"summary shape mismatch: {path}")
        if not all(np.isfinite(array).all() for array in (mean, lower, upper, target)):
            raise RuntimeError(f"non-finite probabilistic summary: {path}")
        if not np.all(lower <= upper):
            raise RuntimeError(f"invalid interval: {path}")
        error = np.abs(mean - target)
        threshold = 10.0 if dataset == "Seattle" else 1.0
        mask = target > threshold
        interval = (upper - lower) + 40.0 * (
            np.maximum(lower - target, 0.0) + np.maximum(target - upper, 0.0)
        )
        canonical = np.rint(target).astype(np.float32, copy=False)
        canonical[canonical == 0] = 0.0
        target_hash = sha256_bytes(np.ascontiguousarray(canonical).tobytes())
        values = {
            "MAE": float(error.mean(dtype=np.float64)),
            "RMSE": float(np.sqrt(np.mean((mean - target) ** 2, dtype=np.float64))),
            "MAPE": float(
                100.0
                * np.sum(error[mask] / np.maximum(np.abs(target[mask]), 1e-5), dtype=np.float64)
                / max(int(mask.sum()), 1)
            ),
            "CRPS": None,
            "MIS": float(interval.mean(dtype=np.float64)),
        }
        return values, target_hash, list(target.shape)


def audit_seed(root: Path, dataset: str, seed: int) -> tuple[dict, dict]:
    seed_root = root / dataset / f"seed{seed}"
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
        raise RuntimeError(f"{dataset} seed{seed} missing/empty artifacts: {missing}")
    metrics = read_json(seed_root / "metrics.json")
    if metrics.get("model") != "TOCU" or metrics.get("model_alias") != "TOCU":
        raise RuntimeError(f"{dataset} seed{seed} model identity mismatch")
    if metrics.get("dataset") != dataset or metrics.get("seed") != seed or metrics.get("status") != "completed":
        raise RuntimeError(f"{dataset} seed{seed} provenance mismatch")
    if metrics.get("protocol_label") not in EXPECTED_PROTOCOLS:
        raise RuntimeError(f"{dataset} seed{seed} protocol mismatch: {metrics.get('protocol_label')!r}")
    if metrics.get("training_mode") != "train_plus_validation_refit_TOCU_integrated":
        raise RuntimeError(f"{dataset} seed{seed} training mode mismatch")
    if tuple(metrics.get("split", ())) != (0.6, 0.2, 0.2) or metrics.get("input_len") != 12 or metrics.get("output_len") != 12:
        raise RuntimeError(f"{dataset} seed{seed} public configuration mismatch")
    if int(metrics.get("num_samples", 50)) != 50:
        raise RuntimeError(f"{dataset} seed{seed} sample count mismatch")
    if not EXPECTED_INNOVATIONS.issubset(set(metrics.get("innovation_flags", []))):
        raise RuntimeError(f"{dataset} seed{seed} innovation flags mismatch")
    history = read_json(seed_root / "history.json")
    for key in ("backbone", "adapter", "history_selection", "head"):
        if key not in history or not history[key]:
            raise RuntimeError(f"{dataset} seed{seed} incomplete training history: {key}")
    # A guarded identity fallback is a valid history-selection outcome.  In
    # that case no history adapter is refit, so the final `history` list is
    # intentionally empty; require the recorded zero epoch/blend instead.
    selected_history_epoch = int(metrics.get("selected_history_epoch", -1))
    selected_history_blend = float(metrics.get("selected_history_blend", float("nan")))
    if selected_history_epoch < 0:
        raise RuntimeError(f"{dataset} seed{seed} invalid selected history epoch")
    if selected_history_epoch == 0:
        if abs(selected_history_blend) > 1e-12:
            raise RuntimeError(f"{dataset} seed{seed} identity history fallback has nonzero blend")
    elif not history.get("history"):
        raise RuntimeError(f"{dataset} seed{seed} incomplete training history: history")
    selection = read_json(seed_root / "selection.json")
    if selection.get("dataset") != dataset or int(selection.get("seed", -1)) != seed:
        raise RuntimeError(f"{dataset} seed{seed} selection provenance mismatch")
    if "test_read" not in str(selection.get("protocol", "")):
        raise RuntimeError(f"{dataset} seed{seed} selection protocol missing test-read guard")
    recomputed, target_hash, target_shape = recompute_summary(seed_root / "summary.npz", dataset)
    if target_hash != metrics.get("target_sha256") or target_shape != metrics.get("target_shape"):
        raise RuntimeError(f"{dataset} seed{seed} target provenance mismatch")
    # CRPS is computed from the full sample path during training and is not
    # recoverable from the stored mean/interval summary alone.
    deltas = {
        name: abs(recomputed[name] - float(metrics[name]))
        for name in ("MAE", "RMSE", "MAPE", "MIS")
    }
    tolerances = {"MAE": 5e-4, "RMSE": 5e-4, "MAPE": 5e-4, "MIS": 5e-3}
    if any(deltas[name] > tolerances[name] for name in deltas):
        raise RuntimeError(f"{dataset} seed{seed} metric mismatch: {deltas}")
    return metrics, {
        "dataset": dataset,
        "seed": seed,
        "target_sha256": target_hash,
        "target_shape": target_shape,
        "recomputed_metric_deltas": deltas,
        "crps_recomputed": False,
    }


def aggregate(rows: list[dict]) -> dict[str, dict[str, float]]:
    return {
        metric: {
            "mean": float(np.mean([float(row[metric]) for row in rows])),
            "sample_std": float(np.std([float(row[metric]) for row in rows], ddof=1)),
        }
        for metric in METRICS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    accepted: dict[str, list[dict]] = {}
    checks: dict[str, list[dict]] = {}
    for dataset in DATASETS:
        accepted[dataset] = []
        checks[dataset] = []
        for seed in SEEDS:
            metrics, check = audit_seed(args.root, dataset, seed)
            accepted[dataset].append(metrics)
            checks[dataset].append(check)
        hashes = {row["target_sha256"] for row in accepted[dataset]}
        if len(hashes) != 1:
            raise RuntimeError(f"{dataset} seed target hashes differ: {hashes}")
    payload = {
        "status": "passed",
        "model": "TOCU",
        "protocol_labels_accepted": sorted(EXPECTED_PROTOCOLS),
        "datasets": DATASETS,
        "seeds": SEEDS,
        "checks": checks,
        "tocu_mean_std": {dataset: aggregate(accepted[dataset]) for dataset in DATASETS},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
