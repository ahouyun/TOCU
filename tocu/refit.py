#!/usr/bin/env python3
"""TOCU: history and ratio calibration selected on one point path.

The formal protocol selects the ratio rule on the history adapter stored in the
source checkpoint, then refit a different history adapter before applying that
rule.  TOCU keeps the validation-selected history blend and support cutoff
fixed, selects the ratio rule on that exact history-corrected train/validation
path, and only then refits both pieces on train+validation.

``--selection-only`` never iterates over the test loader.  A normal run uses
the same validation-only selection and reads test labels only in the final
``evaluate`` call.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import tocu.train as tocu
from tocu.head import build_context, fit_basis, seed_everything
from tocu.refit_support import (
    apply_history,
    build_frozen_adapter,
    collect_frozen_split,
    fit_head_fixed,
    fit_history_fixed,
    fit_pred_bin_horizon_ratio,
    make_args,
    sha256_array,
)
from tocu.history import robust_normalize, robust_statistics, train_history_adapter


METRICS = ("mae", "rmse", "mape")
BIN_COUNTS = (8, 12, 20)
POWERS = (0.5, 1.0, 1.5)
SHRINKS = (0.0, 0.5, 0.75, 0.9)
BLENDS = (0.05, 0.10, 0.15, 0.25, 0.50)
CUTOFF_QUANTILES = (0.70, 0.85, 0.95, 1.0)
BIAS_CUTOFF_QUANTILES = (0.85, 0.95, 1.0)
BIAS_POWERS = (0.0, 0.5)
BIAS_RIDGES = (0.0, 0.5)
BIAS_SHRINKS = (0.75, 0.9)
BIAS_BLENDS = (0.10, 0.25, 0.50)
BIAS_HORIZON_STARTS = (4, 6, 8)
BIAS_CAPS = (50.0, 100.0, 200.0)
# Affine correction is deliberately a separate, registered exploratory
# profile.  It is fitted on train only and shrunk toward identity before any
# validation guard is applied.  The grid is kept small enough for a complete
# three-seed validation sweep on one GPU.
AFFINE_HORIZON_STARTS = (0, 4, 8)
AFFINE_POWERS = (0.0, 0.5)
AFFINE_RIDGES = (0.0, 0.5)
AFFINE_SHRINKS = (0.5, 0.75, 0.9)
AFFINE_BLENDS = (0.10, 0.25, 0.50, 1.0)
AFFINE_CAPS = (50.0, 100.0)
AFFINE_GATE_QUANTILES = (0.25, 0.50, 0.70, 0.85)
# Isolated Seattle RMSE-tail profile.  Negative powers up-weight high-flow
# targets in the train-only affine fit; the released affine profile stays
# byte-for-byte unchanged.
RMSE_TAIL_AFFINE_HORIZON_STARTS = (0, 4, 8)
RMSE_TAIL_AFFINE_POWERS = (-1.0, -0.5, 0.0, 0.5)
RMSE_TAIL_AFFINE_RIDGES = (0.0, 0.5, 2.0)
RMSE_TAIL_AFFINE_SHRINKS = (0.5, 0.75, 0.9)
RMSE_TAIL_AFFINE_BLENDS = (0.10, 0.25, 0.50, 1.0)
RMSE_TAIL_AFFINE_CAPS = (50.0, 100.0, 200.0)
# Conservative tail profile: keep the same train-only tail correction family,
# but exclude the strongest blend and the least-regularized settings.  This
# is a registered validation profile for controlling refit over-correction.
RMSE_TAIL_AFFINE_CONSERVATIVE_HORIZON_STARTS = (0, 4, 8)
RMSE_TAIL_AFFINE_CONSERVATIVE_POWERS = (-0.5, 0.0)
RMSE_TAIL_AFFINE_CONSERVATIVE_RIDGES = (0.0, 0.5)
RMSE_TAIL_AFFINE_CONSERVATIVE_SHRINKS = (0.5, 0.75)
RMSE_TAIL_AFFINE_CONSERVATIVE_BLENDS = (0.10, 0.25, 0.50)
RMSE_TAIL_AFFINE_CONSERVATIVE_CAPS = (50.0, 100.0)
# Ultra-conservative tail profile: keep only strongly regularized affine
# fits and cap the applied blend so the train+validation refit cannot flip
# the signed Seattle bias from negative to positive.
RMSE_TAIL_AFFINE_ULTRA_CONSERVATIVE_HORIZON_STARTS = (0, 4, 8)
RMSE_TAIL_AFFINE_ULTRA_CONSERVATIVE_POWERS = (-0.5, 0.0)
RMSE_TAIL_AFFINE_ULTRA_CONSERVATIVE_RIDGES = (0.5, 2.0)
RMSE_TAIL_AFFINE_ULTRA_CONSERVATIVE_SHRINKS = (0.75, 0.9, 0.95)
RMSE_TAIL_AFFINE_ULTRA_CONSERVATIVE_BLENDS = (0.10, 0.15, 0.25)
RMSE_TAIL_AFFINE_ULTRA_CONSERVATIVE_CAPS = (50.0, 100.0)
# Prediction-value-conditioned affine calibration is kept as a separate,
# validation-only profile.  It adds local slope/intercept flexibility while
# retaining the identity prior and the same three-metric guard.
PRED_AFFINE_BIN_COUNTS = (3, 5, 7)
PRED_AFFINE_POWERS = (0.0, 0.5)
PRED_AFFINE_RIDGES = (0.0, 0.5)
PRED_AFFINE_SHRINKS = (0.5, 0.75, 0.9)
PRED_AFFINE_BLENDS = (0.10, 0.25, 0.50, 1.0)
PRED_AFFINE_CAPS = (50.0, 100.0)
# Per-node affine maps address node-specific level/slope drift.  The strong
# shrink choices are intentional because each map has only train support.
NODE_AFFINE_HORIZON_STARTS = (0, 4, 8)
NODE_AFFINE_POWERS = (0.0, 0.5)
NODE_AFFINE_RIDGES = (0.0, 0.5, 2.0)
NODE_AFFINE_SHRINKS = (0.75, 0.9, 0.95)
NODE_AFFINE_BLENDS = (0.10, 0.25, 0.50, 1.0)
NODE_AFFINE_CAPS = (50.0, 100.0)
# Sparse high-flow node bias profile.  Nodes are selected from train residuals
# only, then corrected additively with strong shrinkage; this avoids fitting a
# free affine map for all 323 nodes.
SPARSE_BIAS_CUTOFF_QUANTILES = (0.85, 0.95)
SPARSE_BIAS_POWERS = (-0.5, 0.0, 0.5)
SPARSE_BIAS_RIDGES = (0.5, 2.0)
SPARSE_BIAS_SHRINKS = (0.5, 0.75, 0.9)
SPARSE_BIAS_BLENDS = (0.25, 0.5, 1.0)
SPARSE_BIAS_HORIZON_STARTS = (4, 8)
SPARSE_BIAS_CAPS = (100.0, 200.0)
SPARSE_BIAS_TOPKS = (4, 8, 16, 32)


def ratio_grid(profile: str) -> tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Return a registered validation-only ratio grid.

    The default profile is byte-for-byte compatible with the released TOCU
    selection path.  ``expanded`` is an explicitly named exploratory profile;
    it widens the train-derived grid without touching test labels.
    """
    if profile == "default":
        return BIN_COUNTS, POWERS, SHRINKS, BLENDS, CUTOFF_QUANTILES
    if profile == "expanded":
        return (
            (4, 8, 12, 16, 20, 32),
            (0.25, 0.5, 0.75, 1.0),
            (0.0, 0.25, 0.5, 0.75, 0.9),
            (0.025, 0.05, 0.10, 0.15, 0.25, 0.50),
            (0.55, 0.70, 0.80, 0.90, 0.95, 1.0),
        )
    raise ValueError(f"unsupported ratio grid profile: {profile}")


def point_metrics(point: torch.Tensor, target: torch.Tensor, dataset: str) -> dict[str, float]:
    residual = point - target
    threshold = 10.0 if dataset == "Seattle" else 1.0
    mask = target > threshold
    return {
        "mae": float(residual.abs().mean()),
        "rmse": float(residual.square().mean().sqrt()),
        "mape": float(
            100.0
            * (residual.abs()[mask] / target.abs()[mask].clamp_min(1e-6)).mean()
        ),
    }


def ratio_hash(calibration: dict | None) -> str | None:
    if calibration is None:
        return None
    return tocu.ratio_payload_sha256(calibration)


def apply_ratio(point: torch.Tensor, row: dict) -> torch.Tensor:
    calibration = row.get("calibration")
    if calibration is None:
        return point
    return tocu.apply_ratio_calibration(point, calibration)


def apply_history_overrides(formal_args, args):
    """Apply explicit CLI overrides after loading the audited source config.

    The source checkpoint remains the default for compatibility.  Overrides
    are opt-in and are recorded in the output provenance so a history sweep
    cannot silently change the TOCU protocol.
    """
    names = (
        "history_hidden",
        "history_node_dim",
        "history_horizon_dim",
        "history_max_delta",
        "history_log_weight",
        "history_relative_weight",
        "history_relative_target_power",
        "history_mae_weight",
        "history_rmse_weight",
        "history_delta_penalty",
        "history_horizon_weight",
        "history_lr",
        "history_weight_decay",
    )
    overrides = {}
    for name in names:
        value = getattr(args, name, None)
        if value is not None:
            current = getattr(formal_args, name)
            cast = int if name in {"history_hidden", "history_node_dim", "history_horizon_dim"} else float
            cast_value = cast(value)
            setattr(formal_args, name, cast_value)
            overrides[name] = cast_value
    return overrides


def public_row(row: dict | None) -> dict | None:
    if row is None:
        return None
    keys = (
        "family", "gate", "bin_count", "cutoff", "power", "ridge", "blend", "shrink",
        "horizon_start", "bias_cap", "top_k", "val", "score", "eligible", "ratio_hash", "edges_hash",
    )
    public = {key: row[key] for key in keys if key in row}
    if public.get("top_k") is None:
        public.pop("top_k", None)
    return public


def _candidate(
    family: str,
    train_point: torch.Tensor,
    train_target: torch.Tensor,
    dataset: str,
    device: torch.device,
    *,
    cutoff: float | None = None,
    power: float | None = None,
    ridge: float = 1.0,
    blend: float = 0.0,
    shrink: float = 0.0,
    bin_count: int | None = None,
    gate: str = "below",
    horizon_start: int | None = None,
    bias_cap: float | None = None,
    top_k: int | None = None,
) -> dict:
    if family == "identity":
        return {"family": "identity", "calibration": None, "blend": 0.0, "shrink": 0.0, "fit_key": ("identity",)}
    if family == "horizon_affine":
        if power is None or horizon_start is None or bias_cap is None:
            raise ValueError("horizon_affine requires power, horizon_start and bias_cap")
        if gate not in {"none", "below", "above"}:
            raise ValueError(f"unsupported horizon_affine gate: {gate}")
        if gate != "none" and cutoff is None:
            raise ValueError("gated horizon_affine requires cutoff")
        raw = tocu.fit_horizon_affine(
            train_point,
            train_target,
            dataset,
            float(power),
            float(ridge),
            horizon_start=int(horizon_start),
            bias_cap=float(bias_cap),
            gate=gate,
            cutoff=cutoff,
        ).detach().cpu()
        final = torch.stack(
            (
                1.0 + (1.0 - float(shrink)) * (raw[:, 0] - 1.0),
                (1.0 - float(shrink)) * raw[:, 1],
            ),
            dim=1,
        )
        calibration = {
            "family": family,
            "gate": gate,
            "cutoff": None if cutoff is None else float(cutoff),
            "relative_power": float(power),
            "ridge": float(ridge),
            "blend": float(blend),
            "shrink": 0.0,
            "fit_shrink": float(shrink),
            "horizon_start": int(horizon_start),
            "bias_cap": float(bias_cap),
            "ratio": final,
        }
    elif family == "horizon_node_affine":
        if power is None or horizon_start is None or bias_cap is None:
            raise ValueError(
                "horizon_node_affine requires power, horizon_start and bias_cap"
            )
        if gate not in {"none", "below", "above"}:
            raise ValueError(f"unsupported horizon_node_affine gate: {gate}")
        if gate != "none" and cutoff is None:
            raise ValueError("gated horizon_node_affine requires cutoff")
        raw = tocu.fit_horizon_node_affine(
            train_point,
            train_target,
            dataset,
            float(power),
            float(ridge),
            horizon_start=int(horizon_start),
            bias_cap=float(bias_cap),
            gate=gate,
            cutoff=cutoff,
        )
        final = torch.stack(
            (
                1.0 + (1.0 - float(shrink)) * (raw[..., 0] - 1.0),
                (1.0 - float(shrink)) * raw[..., 1],
            ),
            dim=-1,
        )
        calibration = {
            "family": family,
            "gate": gate,
            "cutoff": None if cutoff is None else float(cutoff),
            "relative_power": float(power),
            "ridge": float(ridge),
            "blend": float(blend),
            "shrink": 0.0,
            "fit_shrink": float(shrink),
            "horizon_start": int(horizon_start),
            "bias_cap": float(bias_cap),
            "ratio": final,
        }
    elif family == "pred_bin_affine":
        if power is None or bin_count is None or bias_cap is None:
            raise ValueError("pred_bin_affine requires power, bin_count and bias_cap")
        if gate not in {"none", "below", "above"}:
            raise ValueError(f"unsupported pred_bin_affine gate: {gate}")
        if gate != "none" and cutoff is None:
            raise ValueError("gated pred_bin_affine requires cutoff")
        raw, edges = tocu.fit_pred_bin_affine(
            train_point,
            train_target,
            dataset,
            int(bin_count),
            float(power),
            float(ridge),
            device,
            gate=gate,
            cutoff=cutoff,
            bias_cap=float(bias_cap),
        )
        final = torch.stack(
            (
                1.0 + (1.0 - float(shrink)) * (raw[..., 0] - 1.0),
                (1.0 - float(shrink)) * raw[..., 1],
            ),
            dim=-1,
        )
        calibration = {
            "family": family,
            "gate": gate,
            "bin_count": int(bin_count),
            "cutoff": None if cutoff is None else float(cutoff),
            "relative_power": float(power),
            "ridge": float(ridge),
            "blend": float(blend),
            "shrink": 0.0,
            "fit_shrink": float(shrink),
            "bias_cap": float(bias_cap),
            "ratio": final,
            "edges": edges,
        }
    elif cutoff is None or power is None:
        raise ValueError("ratio candidate requires cutoff and power")
    if family not in {"horizon_affine", "horizon_node_affine", "pred_bin_affine"} and gate not in {"below", "above"}:
        raise ValueError(f"unsupported ratio calibration gate: {gate}")
    if family in {"horizon_affine", "horizon_node_affine", "pred_bin_affine"}:
        pass
    elif family == "pred_bin_horizon":
        raw, edges = fit_pred_bin_horizon_ratio(
            train_point,
            train_target,
            dataset,
            int(bin_count),
            float(power),
            float(cutoff),
            float(ridge),
            device,
            gate=gate,
        )
        prior = raw.mean(dim=1, keepdim=True)
        final = (1.0 - float(shrink)) * raw + float(shrink) * prior
        calibration = {
            "family": family,
            "gate": gate,
            "bin_count": int(bin_count),
            "cutoff": float(cutoff),
            "relative_power": float(power),
            "ridge": float(ridge),
            "blend": float(blend),
            # The shrink has already been applied to the ratio tensor.
            "shrink": 0.0,
            "fit_shrink": float(shrink),
            "ratio": final,
            "edges": edges,
        }
    elif family == "horizon_node":
        raw = tocu.fit_ratio_calibration(
            train_point,
            train_target,
            dataset,
            family,
            float(power),
            float(cutoff),
            float(ridge),
            gate=gate,
        ).detach().cpu()
        final = 1.0 + (1.0 - float(shrink)) * (raw - 1.0)
        calibration = {
            "family": family,
            "gate": gate,
            "cutoff": float(cutoff),
            "relative_power": float(power),
            "ridge": float(ridge),
            "blend": float(blend),
            # Materialize shrink into the ratio tensor so the batched
            # evaluator can treat all families identically.
            "shrink": 0.0,
            "fit_shrink": float(shrink),
            "ratio": final,
        }
    elif family == "horizon_node_bias":
        if horizon_start is None or bias_cap is None:
            raise ValueError("horizon_node_bias requires horizon_start and bias_cap")
        raw = tocu.fit_horizon_node_bias(
            train_point,
            train_target,
            dataset,
            float(power),
            float(cutoff),
            float(ridge),
            gate=gate,
            horizon_start=int(horizon_start),
            bias_cap=float(bias_cap),
        ).detach().cpu()
        selected_nodes = None
        if top_k is not None:
            top_k = max(1, min(int(top_k), int(raw.shape[1])))
            node_score = raw.abs().mean(dim=0)
            selected_nodes = torch.topk(node_score, k=top_k, largest=True, sorted=True).indices.tolist()
            node_mask = torch.zeros(raw.shape[1], dtype=raw.dtype)
            node_mask[selected_nodes] = 1.0
            raw = raw * node_mask.reshape(1, -1)
        calibration = {
            "family": family,
            "gate": gate,
            "cutoff": float(cutoff),
            "relative_power": float(power),
            "ridge": float(ridge),
            "blend": float(blend),
            "shrink": 0.0,
            "fit_shrink": float(shrink),
            "horizon_start": int(horizon_start),
            "bias_cap": float(bias_cap),
            "ratio": raw,
        }
        if selected_nodes is not None:
            calibration["selected_nodes"] = [int(node) for node in selected_nodes]
    else:
        raise ValueError(f"unsupported ratio family: {family}")
    row = {
        "family": family,
        "gate": gate,
        "bin_count": int(bin_count) if bin_count is not None else None,
        "cutoff": None if cutoff is None else float(cutoff),
        "power": float(power),
        "ridge": float(ridge),
        "blend": float(blend),
        "shrink": float(shrink),
        "horizon_start": int(horizon_start) if horizon_start is not None else None,
        "bias_cap": float(bias_cap) if bias_cap is not None else None,
        "top_k": int(top_k) if top_k is not None else None,
        "calibration": calibration,
        "ratio_hash": ratio_hash(calibration),
        "fit_key": (
            family,
            gate,
            int(bin_count) if bin_count is not None else None,
            None if cutoff is None else float(cutoff),
            float(power),
            float(ridge),
            int(horizon_start) if horizon_start is not None else None,
            float(bias_cap) if bias_cap is not None else None,
            int(top_k) if top_k is not None else None,
        ),
    }
    if calibration.get("edges") is not None:
        row["edges_hash"] = sha256_array(calibration["edges"].numpy())
    return row


def ratio_variant(base: dict, blend: float, shrink: float) -> dict:
    """Materialize a blend/shrink variant from one already-fitted basis."""
    if base["family"] == "identity":
        return dict(base)
    calibration = dict(base["calibration"])
    raw = base["calibration"]["ratio"]
    if base["family"] == "pred_bin_horizon":
        prior = raw.mean(dim=1, keepdim=True)
        final = (1.0 - float(shrink)) * raw + float(shrink) * prior
    elif base["family"] == "horizon_node_bias":
        prior = raw.mean(dim=1, keepdim=True)
        selected_nodes = base["calibration"].get("selected_nodes")
        if selected_nodes is not None:
            node_mask = torch.zeros(raw.shape[1], dtype=raw.dtype)
            node_mask[[int(node) for node in selected_nodes]] = 1.0
            prior = prior * node_mask.reshape(1, -1)
        final = (1.0 - float(shrink)) * raw + float(shrink) * prior
    elif base["family"] == "horizon_affine":
        final = torch.stack(
            (
                1.0 + (1.0 - float(shrink)) * (raw[:, 0] - 1.0),
                (1.0 - float(shrink)) * raw[:, 1],
            ),
            dim=1,
        )
    elif base["family"] == "pred_bin_affine":
        final = torch.stack(
            (
                1.0 + (1.0 - float(shrink)) * (raw[..., 0] - 1.0),
                (1.0 - float(shrink)) * raw[..., 1],
            ),
            dim=-1,
        )
    elif base["family"] == "horizon_node_affine":
        final = torch.stack(
            (
                1.0 + (1.0 - float(shrink)) * (raw[..., 0] - 1.0),
                (1.0 - float(shrink)) * raw[..., 1],
            ),
            dim=-1,
        )
    else:
        final = 1.0 + (1.0 - float(shrink)) * (raw - 1.0)
    calibration["ratio"] = final
    calibration["blend"] = float(blend)
    calibration["shrink"] = 0.0
    calibration["fit_shrink"] = float(shrink)
    row = dict(base)
    row.update({"blend": float(blend), "shrink": float(shrink), "calibration": calibration})
    row["ratio_hash"] = ratio_hash(calibration)
    return row


def evaluate_ratio_rows(
    prediction: torch.Tensor,
    target: torch.Tensor,
    rows: list[dict],
    dataset: str,
    device: torch.device,
    row_chunk: int = 24,
    sample_chunk: int = 256,
) -> None:
    """Evaluate ratio candidates in batched GPU blocks.

    Rows sharing a fitted ratio/edge basis are evaluated together.  This
    avoids the probe's pathological ``candidate x full_tensor`` loop while
    preserving the exact MAE/RMSE/MAPE reductions used by the audited probes.
    """
    threshold = 10.0 if dataset == "Seattle" else 1.0
    p_all = prediction.to(device=device, dtype=torch.float32)
    t_all = target.to(device=device, dtype=torch.float32)
    valid_all = t_all > float(threshold)
    total = float(target.numel())
    valid_count = float(valid_all.sum().item())
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get("fit_key", (row["family"],))), []).append(row)

    for fit_key, group in groups.items():
        family = str(group[0]["family"])
        for start in range(0, len(group), int(row_chunk)):
            chunk = group[start : start + int(row_chunk)]
            count = len(chunk)
            if family == "identity":
                scales = None
                edges = None
                cutoff = float("inf")
                gate_mode = "below"
            else:
                scales = torch.stack(
                    [row["calibration"]["ratio"] for row in chunk], dim=0
                ).to(device=device, dtype=torch.float32)
                edges = chunk[0]["calibration"].get("edges")
                if edges is not None:
                    edges = edges.to(device=device, dtype=torch.float32)
                if family in {"horizon_affine", "horizon_node_affine", "pred_bin_affine"}:
                    raw_cutoff = chunk[0]["calibration"].get("cutoff")
                    cutoff = float("inf") if raw_cutoff is None else float(raw_cutoff)
                    gate_mode = str(chunk[0]["calibration"].get("gate", "none"))
                else:
                    cutoff = float(chunk[0]["cutoff"])
                    gate_mode = str(chunk[0]["calibration"].get("gate", "below"))
            blends = torch.as_tensor(
                [float(row.get("blend", 0.0)) for row in chunk],
                device=device,
                dtype=torch.float32,
            ).reshape(count, 1, 1, 1)
            abs_sum = torch.zeros(count, device=device, dtype=torch.float64)
            sq_sum = torch.zeros(count, device=device, dtype=torch.float64)
            rel_sum = torch.zeros(count, device=device, dtype=torch.float64)
            for sample_start in range(0, p_all.shape[0], int(sample_chunk)):
                stop = min(sample_start + int(sample_chunk), p_all.shape[0])
                p = p_all[sample_start:stop]
                t = t_all[sample_start:stop]
                valid = valid_all[sample_start:stop]
                if family == "identity":
                    corrected = p.unsqueeze(0).expand(count, -1, -1, -1)
                elif family == "horizon_affine":
                    params = scales
                    slope = params[:, :, 0].reshape(count, 1, -1, 1)
                    intercept = params[:, :, 1].reshape(count, 1, -1, 1)
                    correction = ((slope - 1.0) * p.unsqueeze(0) + intercept)
                    correction = correction.clamp(
                        min=-float(chunk[0]["calibration"].get("bias_cap", 100.0)),
                        max=float(chunk[0]["calibration"].get("bias_cap", 100.0)),
                    )
                    corrected = p.unsqueeze(0) + blends * correction
                    horizon_start = int(chunk[0]["calibration"].get("horizon_start", 0))
                    horizon_mask = (
                        torch.arange(p.shape[1], device=device)
                        .view(1, 1, -1, 1)
                        >= horizon_start
                    )
                    if gate_mode == "below":
                        gate = p.unsqueeze(0) < cutoff
                    elif gate_mode == "above":
                        gate = p.unsqueeze(0) >= cutoff
                    elif gate_mode == "none":
                        gate = torch.ones_like(p.unsqueeze(0), dtype=torch.bool)
                    else:
                        raise ValueError(f"unsupported horizon affine gate: {gate_mode}")
                    corrected = torch.where(
                        horizon_mask & gate,
                        corrected.clamp_min(0.0),
                        p.unsqueeze(0),
                    )
                elif family == "horizon_node_affine":
                    params = scales
                    slope = params[:, :, :, 0].reshape(count, 1, p.shape[1], p.shape[2])
                    intercept = params[:, :, :, 1].reshape(count, 1, p.shape[1], p.shape[2])
                    correction = (slope - 1.0) * p.unsqueeze(0) + intercept
                    correction = correction.clamp(
                        min=-float(chunk[0]["calibration"].get("bias_cap", 100.0)),
                        max=float(chunk[0]["calibration"].get("bias_cap", 100.0)),
                    )
                    corrected = p.unsqueeze(0) + blends * correction
                    horizon_start = int(chunk[0]["calibration"].get("horizon_start", 0))
                    horizon_mask = (
                        torch.arange(p.shape[1], device=device)
                        .view(1, 1, -1, 1)
                        >= horizon_start
                    )
                    if gate_mode == "below":
                        gate = p.unsqueeze(0) < cutoff
                    elif gate_mode == "above":
                        gate = p.unsqueeze(0) >= cutoff
                    elif gate_mode == "none":
                        gate = torch.ones_like(p.unsqueeze(0), dtype=torch.bool)
                    else:
                        raise ValueError(f"unsupported horizon-node affine gate: {gate_mode}")
                    corrected = torch.where(
                        horizon_mask & gate,
                        corrected.clamp_min(0.0),
                        p.unsqueeze(0),
                    )
                elif family == "horizon_node_bias":
                    bias = scales[:, None, :, :]
                    corrected = p.unsqueeze(0) + blends * bias
                    horizon_start = int(chunk[0]["calibration"].get("horizon_start", 0))
                    horizon_mask = (
                        torch.arange(p.shape[1], device=device)
                        .view(1, 1, -1, 1)
                        >= horizon_start
                    )
                    if gate_mode == "below":
                        gate = p.unsqueeze(0) < cutoff
                    elif gate_mode == "above":
                        gate = p.unsqueeze(0) >= cutoff
                    else:
                        raise ValueError(f"unsupported ratio calibration gate: {gate_mode}")
                    corrected = torch.where(
                        gate & horizon_mask,
                        corrected.clamp_min(0.0),
                        p.unsqueeze(0),
                    )
                elif family == "horizon_node":
                    factor = scales[:, None, :, :]
                    corrected = p.unsqueeze(0) + blends * (
                        p.unsqueeze(0) * factor - p.unsqueeze(0)
                    )
                    if gate_mode == "below":
                        gate = p.unsqueeze(0) < cutoff
                    elif gate_mode == "above":
                        gate = p.unsqueeze(0) >= cutoff
                    else:
                        raise ValueError(f"unsupported ratio calibration gate: {gate_mode}")
                    corrected = torch.where(gate, corrected.clamp_min(0.0), p.unsqueeze(0))
                elif family == "pred_bin_affine":
                    bins = []
                    for horizon in range(p.shape[1]):
                        bins.append(
                            torch.bucketize(
                                p[:, horizon].contiguous(),
                                edges[horizon, 1:-1],
                                right=True,
                            )
                        )
                    bin_tensor = torch.stack(bins, dim=1)
                    selected = []
                    for horizon in range(p.shape[1]):
                        params = scales[:, horizon]
                        index = bin_tensor[:, horizon].unsqueeze(0).unsqueeze(-1)
                        index = index.expand(count, -1, -1, 2)
                        selected.append(
                            torch.gather(
                                params.unsqueeze(1).expand(count, p.shape[0], -1, 2),
                                2,
                                index,
                            )
                        )
                    params = torch.stack(selected, dim=2)
                    slope = params[..., 0]
                    intercept = params[..., 1]
                    calibrated = slope * p.unsqueeze(0) + intercept
                    corrected = p.unsqueeze(0) + blends * (calibrated - p.unsqueeze(0))
                    if gate_mode == "below":
                        gate = p.unsqueeze(0) < cutoff
                    elif gate_mode == "above":
                        gate = p.unsqueeze(0) >= cutoff
                    elif gate_mode == "none":
                        gate = torch.ones_like(p.unsqueeze(0), dtype=torch.bool)
                    else:
                        raise ValueError(f"unsupported prediction-bin affine gate: {gate_mode}")
                    corrected = torch.where(gate, corrected.clamp_min(0.0), p.unsqueeze(0))
                elif family == "pred_bin_horizon":
                    # Bin assignment is shared by all blend/shrink variants
                    # in this fitted group.
                    bins = []
                    for horizon in range(p.shape[1]):
                        bins.append(
                            torch.bucketize(
                                p[:, horizon].contiguous(),
                                edges[horizon, 1:-1],
                                right=True,
                            )
                        )
                    bin_tensor = torch.stack(bins, dim=1)
                    gathered = []
                    for horizon in range(p.shape[1]):
                        ratio_h = scales[:, horizon, :]
                        index_h = bin_tensor[:, horizon].unsqueeze(0).expand(count, -1, -1)
                        gathered.append(torch.gather(ratio_h[:, None, :].expand(count, p.shape[0], -1), 2, index_h))
                    factor = torch.stack(gathered, dim=2)
                    corrected = p.unsqueeze(0) + blends * (
                        p.unsqueeze(0) * factor - p.unsqueeze(0)
                    )
                    if gate_mode == "below":
                        gate = p.unsqueeze(0) < cutoff
                    elif gate_mode == "above":
                        gate = p.unsqueeze(0) >= cutoff
                    else:
                        raise ValueError(f"unsupported ratio calibration gate: {gate_mode}")
                    corrected = torch.where(gate, corrected.clamp_min(0.0), p.unsqueeze(0))
                else:
                    raise ValueError(family)
                residual = corrected - t.unsqueeze(0)
                absolute = residual.abs()
                abs_sum += absolute.sum((1, 2, 3), dtype=torch.float64)
                sq_sum += residual.square().sum((1, 2, 3), dtype=torch.float64)
                rel_sum += (
                    absolute
                    / t.abs().clamp_min(1e-6).unsqueeze(0)
                    * valid.unsqueeze(0)
                ).sum((1, 2, 3), dtype=torch.float64)
            for index, row in enumerate(chunk):
                row["val"] = {
                    "mae": float((abs_sum[index] / total).cpu()),
                    "rmse": float((sq_sum[index] / total).sqrt().cpu()),
                    "mape": float((100.0 * rel_sum[index] / valid_count).cpu()),
                }
            del scales, edges, blends, abs_sum, sq_sum, rel_sum
            if device.type == "cuda":
                torch.cuda.empty_cache()


def scan_node_bias(
    train_point: torch.Tensor,
    train_target: torch.Tensor,
    val_point: torch.Tensor,
    val_target: torch.Tensor,
    dataset: str,
    device: torch.device,
    guard_tolerance: float,
    selection: str,
    tail_mode: str,
    sparse: bool = False,
) -> tuple[dict, dict]:
    """Scan a constrained additive node-horizon residual gate on validation."""
    baseline = point_metrics(val_point, val_target, dataset)
    threshold = 10.0 if dataset == "Seattle" else 1.0
    values = train_point[train_target > threshold].reshape(-1)
    if values.numel() > 2_000_000:
        stride = max(1, values.numel() // 2_000_000)
        values = values[::stride][:2_000_000]
    cutoff_quantiles = (
        SPARSE_BIAS_CUTOFF_QUANTILES if sparse else BIAS_CUTOFF_QUANTILES
    )
    powers = SPARSE_BIAS_POWERS if sparse else BIAS_POWERS
    ridges = SPARSE_BIAS_RIDGES if sparse else BIAS_RIDGES
    shrinks = SPARSE_BIAS_SHRINKS if sparse else BIAS_SHRINKS
    blends = SPARSE_BIAS_BLENDS if sparse else BIAS_BLENDS
    horizon_starts = SPARSE_BIAS_HORIZON_STARTS if sparse else BIAS_HORIZON_STARTS
    caps = SPARSE_BIAS_CAPS if sparse else BIAS_CAPS
    top_ks = SPARSE_BIAS_TOPKS if sparse else (None,)
    cutoffs = sorted({float(torch.quantile(values, q)) for q in cutoff_quantiles})
    if tail_mode not in {"low", "upper", "both"}:
        raise ValueError(f"unsupported ratio tail mode: {tail_mode}")
    gates = (
        ("above",)
        if sparse
        else (
        ("below",)
        if tail_mode == "low"
        else (("above",) if tail_mode == "upper" else ("below", "above"))
        )
    )
    rows = [_candidate("identity", train_point, train_target, dataset, device)]
    fit_point = train_point.to(device=device, dtype=torch.float32)
    fit_target = train_target.to(device=device, dtype=torch.float32)
    for gate in gates:
        for cutoff in cutoffs:
            for power in powers:
                for ridge in ridges:
                    for horizon_start in horizon_starts:
                        for bias_cap in caps:
                            for top_k in top_ks:
                                base = _candidate(
                                    "horizon_node_bias",
                                    fit_point,
                                    fit_target,
                                    dataset,
                                    device,
                                    cutoff=cutoff,
                                    power=power,
                                    ridge=ridge,
                                    blend=0.0,
                                    shrink=0.0,
                                    gate=gate,
                                    horizon_start=horizon_start,
                                    bias_cap=bias_cap,
                                    top_k=top_k,
                                )
                                for shrink in shrinks:
                                    for blend in blends:
                                        rows.append(ratio_variant(base, blend, shrink))
    evaluate_ratio_rows(val_point, val_target, rows, dataset, device)
    for row in rows:
        row["score"] = sum(row["val"][key] / baseline[key] for key in METRICS)
        row["eligible"] = all(
            row["val"][key] <= baseline[key] * (1.0 + float(guard_tolerance))
            for key in METRICS
        )
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise RuntimeError("node-bias scan produced no guard-eligible identity/candidate")
    if selection == "mape":
        selected = min(eligible, key=lambda row: (row["val"]["mape"], row["val"]["rmse"], row["score"]))
    elif selection == "rmse":
        selected = min(eligible, key=lambda row: (row["val"]["rmse"], row["val"]["mape"], row["score"]))
    else:
        selected = min(eligible, key=lambda row: (row["score"], row["val"]["mape"]))
    def selection_key(row: dict) -> tuple[float, float, float]:
        primary = float(row["score"]) if selection == "score" else float(row["val"][selection])
        return primary, float(row["val"]["mape"]), float(row["score"])
    summary = {
        "selection_protocol": (
            "tocu_history_selected_then_sparse_node_bias_train_fit_validation_guard_no_test"
            if sparse
            else "tocu_history_selected_then_ratio_train_fit_validation_guard_no_test"
        ),
        "tail_mode": tail_mode,
        "grid_profile": "sparse_node_bias" if sparse else "node_bias",
        "grid": {
            "cutoff_quantiles": list(cutoff_quantiles),
            "powers": list(powers),
            "ridges": list(ridges),
            "shrinks": list(shrinks),
            "blends": list(blends),
            "horizon_starts": list(horizon_starts),
            "bias_caps": list(caps),
            "top_ks": [int(value) for value in top_ks if value is not None],
            "gates": list(gates),
        },
        "baseline_validation": baseline,
        "candidate_count": len(rows),
        "eligible_count": len(eligible),
        "selected_by": selection,
        "selected": public_row(selected),
        "top_validation_by_selection": [
            public_row(row) for row in sorted(eligible, key=selection_key)[:20]
        ],
    }
    return selected, summary


def scan_horizon_affine(
    train_point: torch.Tensor,
    train_target: torch.Tensor,
    val_point: torch.Tensor,
    val_target: torch.Tensor,
    dataset: str,
    device: torch.device,
    guard_tolerance: float,
    selection: str,
    tail_mode: str,
    gated: bool = False,
    profile: str = "horizon_affine",
) -> tuple[dict, dict]:
    """Scan a train-only horizon affine correction under validation guards.

    This profile is intentionally isolated from the released ratio grids.  It
    corrects a signed slope/intercept drift across the forecast horizon, so it
    can address low-flow over-prediction and high-flow under-prediction in one
    rule.  No test tensor is touched here.
    """
    baseline = point_metrics(val_point, val_target, dataset)
    fit_point = train_point.to(device=device, dtype=torch.float32)
    fit_target = train_target.to(device=device, dtype=torch.float32)
    rows = [_candidate("identity", train_point, train_target, dataset, device)]
    threshold = 10.0 if dataset == "Seattle" else 1.0
    support_values = fit_point[fit_target > threshold].reshape(-1)
    if support_values.numel() > 2_000_000:
        stride = max(1, support_values.numel() // 2_000_000)
        support_values = support_values[::stride][:2_000_000]
    if gated:
        cutoff_specs = [
            (gate, float(torch.quantile(support_values, q)))
            for gate in ("below", "above")
            for q in AFFINE_GATE_QUANTILES
        ]
    else:
        cutoff_specs = [("none", None)]
    if profile == "rmse_tail_affine":
        horizon_starts = RMSE_TAIL_AFFINE_HORIZON_STARTS
        powers = RMSE_TAIL_AFFINE_POWERS
        ridges = RMSE_TAIL_AFFINE_RIDGES
        shrinks = RMSE_TAIL_AFFINE_SHRINKS
        blends = RMSE_TAIL_AFFINE_BLENDS
        caps = RMSE_TAIL_AFFINE_CAPS
    elif profile == "rmse_tail_affine_conservative":
        horizon_starts = RMSE_TAIL_AFFINE_CONSERVATIVE_HORIZON_STARTS
        powers = RMSE_TAIL_AFFINE_CONSERVATIVE_POWERS
        ridges = RMSE_TAIL_AFFINE_CONSERVATIVE_RIDGES
        shrinks = RMSE_TAIL_AFFINE_CONSERVATIVE_SHRINKS
        blends = RMSE_TAIL_AFFINE_CONSERVATIVE_BLENDS
        caps = RMSE_TAIL_AFFINE_CONSERVATIVE_CAPS
    elif profile == "rmse_tail_affine_ultra_conservative":
        horizon_starts = RMSE_TAIL_AFFINE_ULTRA_CONSERVATIVE_HORIZON_STARTS
        powers = RMSE_TAIL_AFFINE_ULTRA_CONSERVATIVE_POWERS
        ridges = RMSE_TAIL_AFFINE_ULTRA_CONSERVATIVE_RIDGES
        shrinks = RMSE_TAIL_AFFINE_ULTRA_CONSERVATIVE_SHRINKS
        blends = RMSE_TAIL_AFFINE_ULTRA_CONSERVATIVE_BLENDS
        caps = RMSE_TAIL_AFFINE_ULTRA_CONSERVATIVE_CAPS
    elif profile == "horizon_affine":
        horizon_starts = AFFINE_HORIZON_STARTS
        powers = AFFINE_POWERS
        ridges = AFFINE_RIDGES
        shrinks = AFFINE_SHRINKS
        blends = AFFINE_BLENDS
        caps = AFFINE_CAPS
    else:
        raise ValueError(f"unsupported horizon affine profile: {profile}")
    for gate, cutoff in cutoff_specs:
        for horizon_start in horizon_starts:
            for power in powers:
                for ridge in ridges:
                    for bias_cap in caps:
                        base = _candidate(
                            "horizon_affine",
                            fit_point,
                            fit_target,
                            dataset,
                            device,
                            cutoff=cutoff,
                            power=power,
                            ridge=ridge,
                            blend=0.0,
                            shrink=0.0,
                            gate=gate,
                            horizon_start=horizon_start,
                            bias_cap=bias_cap,
                        )
                        for shrink in shrinks:
                            rows.extend(
                                ratio_variant(base, blend, shrink)
                                for blend in blends
                            )
    evaluate_ratio_rows(val_point, val_target, rows, dataset, device)
    for row in rows:
        row["score"] = sum(row["val"][key] / baseline[key] for key in METRICS)
        row["eligible"] = all(
            row["val"][key] <= baseline[key] * (1.0 + float(guard_tolerance))
            for key in METRICS
        )
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise RuntimeError("horizon affine scan produced no guard-eligible identity/candidate")
    if selection == "mape":
        selected = min(eligible, key=lambda row: (row["val"]["mape"], row["val"]["rmse"], row["score"]))
    elif selection == "rmse":
        selected = min(eligible, key=lambda row: (row["val"]["rmse"], row["val"]["mape"], row["score"]))
    else:
        selected = min(eligible, key=lambda row: (row["score"], row["val"]["mape"]))

    def selection_key(row: dict) -> tuple[float, float, float]:
        primary = float(row["score"]) if selection == "score" else float(row["val"][selection])
        return primary, float(row["val"]["mape"]), float(row["score"])

    summary = {
        "selection_protocol": (
            "tocu_history_selected_then_rmse_tail_affine_conservative_train_fit_validation_guard_no_test"
            if profile == "rmse_tail_affine_conservative"
            else "tocu_history_selected_then_rmse_tail_affine_ultra_conservative_train_fit_validation_guard_no_test"
            if profile == "rmse_tail_affine_ultra_conservative"
            else "tocu_history_selected_then_rmse_tail_affine_train_fit_validation_guard_no_test"
            if profile == "rmse_tail_affine"
            else "tocu_history_selected_then_horizon_affine_train_fit_validation_guard_no_test"
        ),
        "tail_mode": tail_mode,
        "grid_profile": (
            "rmse_tail_affine_conservative"
            if profile == "rmse_tail_affine_conservative"
            else "rmse_tail_affine_ultra_conservative"
            if profile == "rmse_tail_affine_ultra_conservative"
            else "rmse_tail_affine"
            if profile == "rmse_tail_affine"
            else "horizon_affine_gated" if gated else "horizon_affine"
        ),
        "grid": {
            "horizon_starts": list(horizon_starts),
            "powers": list(powers),
            "ridges": list(ridges),
            "shrinks": list(shrinks),
            "blends": list(blends),
            "bias_caps": list(caps),
            "gate_quantiles": list(AFFINE_GATE_QUANTILES) if gated else [],
            "gates": ["below", "above"] if gated else ["none"],
        },
        "baseline_validation": baseline,
        "candidate_count": len(rows),
        "eligible_count": len(eligible),
        "selected_by": selection,
        "selected": public_row(selected),
        "top_validation_by_selection": [
            public_row(row) for row in sorted(eligible, key=selection_key)[:20]
        ],
    }
    return selected, summary


def scan_pred_bin_affine(
    train_point: torch.Tensor,
    train_target: torch.Tensor,
    val_point: torch.Tensor,
    val_target: torch.Tensor,
    dataset: str,
    device: torch.device,
    guard_tolerance: float,
    selection: str,
    tail_mode: str,
) -> tuple[dict, dict]:
    """Scan train-fitted prediction-bin affine maps under validation guards."""
    if tail_mode not in {"low", "upper", "both"}:
        raise ValueError(f"unsupported ratio tail mode: {tail_mode}")
    baseline = point_metrics(val_point, val_target, dataset)
    rows = [_candidate("identity", train_point, train_target, dataset, device)]
    fit_point = train_point.to(device=device, dtype=torch.float32)
    fit_target = train_target.to(device=device, dtype=torch.float32)
    # The first candidate is deliberately ungated.  The prediction bins
    # already provide a state-conditioned map; adding a validation-selected
    # absolute cutoff here would multiply the selection degrees of freedom.
    for bin_count in PRED_AFFINE_BIN_COUNTS:
        for power in PRED_AFFINE_POWERS:
            for ridge in PRED_AFFINE_RIDGES:
                for bias_cap in PRED_AFFINE_CAPS:
                    base = _candidate(
                        "pred_bin_affine",
                        fit_point,
                        fit_target,
                        dataset,
                        device,
                        cutoff=None,
                        power=power,
                        ridge=ridge,
                        blend=0.0,
                        shrink=0.0,
                        bin_count=bin_count,
                        gate="none",
                        bias_cap=bias_cap,
                    )
                    for shrink in PRED_AFFINE_SHRINKS:
                        rows.extend(
                            ratio_variant(base, blend, shrink)
                            for blend in PRED_AFFINE_BLENDS
                        )
    evaluate_ratio_rows(val_point, val_target, rows, dataset, device)
    for row in rows:
        row["score"] = sum(row["val"][key] / baseline[key] for key in METRICS)
        row["eligible"] = all(
            row["val"][key] <= baseline[key] * (1.0 + float(guard_tolerance))
            for key in METRICS
        )
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise RuntimeError("prediction-bin affine scan produced no guard-eligible candidate")
    if selection == "mape":
        selected = min(eligible, key=lambda row: (row["val"]["mape"], row["val"]["rmse"], row["score"]))
    elif selection == "rmse":
        selected = min(eligible, key=lambda row: (row["val"]["rmse"], row["val"]["mape"], row["score"]))
    else:
        selected = min(eligible, key=lambda row: (row["score"], row["val"]["mape"]))

    def selection_key(row: dict) -> tuple[float, float, float]:
        primary = float(row["score"]) if selection == "score" else float(row["val"][selection])
        return primary, float(row["val"]["mape"]), float(row["score"])

    summary = {
        "selection_protocol": "tocu_history_selected_then_prediction_bin_affine_train_fit_validation_guard_no_test",
        "tail_mode": tail_mode,
        "grid_profile": "pred_bin_affine",
        "grid": {
            "bin_counts": list(PRED_AFFINE_BIN_COUNTS),
            "powers": list(PRED_AFFINE_POWERS),
            "ridges": list(PRED_AFFINE_RIDGES),
            "shrinks": list(PRED_AFFINE_SHRINKS),
            "blends": list(PRED_AFFINE_BLENDS),
            "bias_caps": list(PRED_AFFINE_CAPS),
            "gates": ["none"],
        },
        "baseline_validation": baseline,
        "candidate_count": len(rows),
        "eligible_count": len(eligible),
        "selected_by": selection,
        "selected": public_row(selected),
        "top_validation_by_selection": [
            public_row(row) for row in sorted(eligible, key=selection_key)[:20]
        ],
    }
    return selected, summary


def scan_horizon_node_affine(
    train_point: torch.Tensor,
    train_target: torch.Tensor,
    val_point: torch.Tensor,
    val_target: torch.Tensor,
    dataset: str,
    device: torch.device,
    guard_tolerance: float,
    selection: str,
    tail_mode: str,
) -> tuple[dict, dict]:
    """Scan train-fitted per-horizon/node affine maps under guards."""
    if tail_mode not in {"low", "upper", "both"}:
        raise ValueError(f"unsupported ratio tail mode: {tail_mode}")
    baseline = point_metrics(val_point, val_target, dataset)
    rows = [_candidate("identity", train_point, train_target, dataset, device)]
    fit_point = train_point.to(device=device, dtype=torch.float32)
    fit_target = train_target.to(device=device, dtype=torch.float32)
    # This first profile is ungated.  The horizon start and strong shrink grid
    # constrain the high-dimensional node maps; a gated variant can be added
    # only after this family proves stable across seeds.
    for horizon_start in NODE_AFFINE_HORIZON_STARTS:
        for power in NODE_AFFINE_POWERS:
            for ridge in NODE_AFFINE_RIDGES:
                for bias_cap in NODE_AFFINE_CAPS:
                    base = _candidate(
                        "horizon_node_affine",
                        fit_point,
                        fit_target,
                        dataset,
                        device,
                        cutoff=None,
                        power=power,
                        ridge=ridge,
                        blend=0.0,
                        shrink=0.0,
                        gate="none",
                        horizon_start=horizon_start,
                        bias_cap=bias_cap,
                    )
                    for shrink in NODE_AFFINE_SHRINKS:
                        rows.extend(
                            ratio_variant(base, blend, shrink)
                            for blend in NODE_AFFINE_BLENDS
                        )
    evaluate_ratio_rows(val_point, val_target, rows, dataset, device)
    for row in rows:
        row["score"] = sum(row["val"][key] / baseline[key] for key in METRICS)
        row["eligible"] = all(
            row["val"][key] <= baseline[key] * (1.0 + float(guard_tolerance))
            for key in METRICS
        )
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise RuntimeError("horizon-node affine scan produced no guard-eligible candidate")
    if selection == "mape":
        selected = min(eligible, key=lambda row: (row["val"]["mape"], row["val"]["rmse"], row["score"]))
    elif selection == "rmse":
        selected = min(eligible, key=lambda row: (row["val"]["rmse"], row["val"]["mape"], row["score"]))
    else:
        selected = min(eligible, key=lambda row: (row["score"], row["val"]["mape"]))

    def selection_key(row: dict) -> tuple[float, float, float]:
        primary = float(row["score"]) if selection == "score" else float(row["val"][selection])
        return primary, float(row["val"]["mape"]), float(row["score"])

    summary = {
        "selection_protocol": "tocu_history_selected_then_horizon_node_affine_train_fit_validation_guard_no_test",
        "tail_mode": tail_mode,
        "grid_profile": "horizon_node_affine",
        "grid": {
            "horizon_starts": list(NODE_AFFINE_HORIZON_STARTS),
            "powers": list(NODE_AFFINE_POWERS),
            "ridges": list(NODE_AFFINE_RIDGES),
            "shrinks": list(NODE_AFFINE_SHRINKS),
            "blends": list(NODE_AFFINE_BLENDS),
            "bias_caps": list(NODE_AFFINE_CAPS),
            "gates": ["none"],
        },
        "baseline_validation": baseline,
        "candidate_count": len(rows),
        "eligible_count": len(eligible),
        "selected_by": selection,
        "selected": public_row(selected),
        "top_validation_by_selection": [
            public_row(row) for row in sorted(eligible, key=selection_key)[:20]
        ],
    }
    return selected, summary


def scan_ratio(
    train_point: torch.Tensor,
    train_target: torch.Tensor,
    val_point: torch.Tensor,
    val_target: torch.Tensor,
    dataset: str,
    device: torch.device,
    guard_tolerance: float,
    selection: str,
    tail_mode: str = "low",
    grid_profile: str = "default",
) -> tuple[dict, dict]:
    """Fit on train and select under a three-metric validation guard."""
    if grid_profile == "horizon_affine":
        return scan_horizon_affine(
            train_point,
            train_target,
            val_point,
            val_target,
            dataset,
            device,
            guard_tolerance,
            selection,
            tail_mode,
        )
    if grid_profile in {
        "rmse_tail_affine",
        "rmse_tail_affine_conservative",
        "rmse_tail_affine_ultra_conservative",
    }:
        return scan_horizon_affine(
            train_point,
            train_target,
            val_point,
            val_target,
            dataset,
            device,
            guard_tolerance,
            selection,
            tail_mode,
            profile=grid_profile,
        )
    if grid_profile == "horizon_affine_gated":
        return scan_horizon_affine(
            train_point,
            train_target,
            val_point,
            val_target,
            dataset,
            device,
            guard_tolerance,
            selection,
            tail_mode,
            gated=True,
        )
    if grid_profile == "pred_bin_affine":
        return scan_pred_bin_affine(
            train_point,
            train_target,
            val_point,
            val_target,
            dataset,
            device,
            guard_tolerance,
            selection,
            tail_mode,
        )
    if grid_profile == "horizon_node_affine":
        return scan_horizon_node_affine(
            train_point,
            train_target,
            val_point,
            val_target,
            dataset,
            device,
            guard_tolerance,
            selection,
            tail_mode,
        )
    if grid_profile == "node_bias":
        return scan_node_bias(
            train_point,
            train_target,
            val_point,
            val_target,
            dataset,
            device,
            guard_tolerance,
            selection,
            tail_mode,
        )
    if grid_profile == "sparse_node_bias":
        return scan_node_bias(
            train_point,
            train_target,
            val_point,
            val_target,
            dataset,
            device,
            guard_tolerance,
            selection,
            tail_mode,
            sparse=True,
        )
    baseline = point_metrics(val_point, val_target, dataset)
    threshold = 10.0 if dataset == "Seattle" else 1.0
    values = train_point[train_target > threshold].reshape(-1)
    if values.numel() > 2_000_000:
        stride = max(1, values.numel() // 2_000_000)
        values = values[::stride][:2_000_000]
    bin_counts, powers, shrinks, blends, cutoff_quantiles = ratio_grid(grid_profile)
    cutoffs = sorted({float(torch.quantile(values, q)) for q in cutoff_quantiles})
    # Keep the full train tensors resident on the selected accelerator.  The
    # Candidate variants reuse the fitted basis instead of copying it repeatedly.
    fit_point = train_point.to(device=device, dtype=torch.float32)
    fit_target = train_target.to(device=device, dtype=torch.float32)
    if tail_mode not in {"low", "upper", "both"}:
        raise ValueError(f"unsupported ratio tail mode: {tail_mode}")
    gates = (
        ("below",)
        if tail_mode == "low"
        else (("above",) if tail_mode == "upper" else ("below", "above"))
    )
    rows = [_candidate("identity", train_point, train_target, dataset, device)]
    for gate in gates:
        for cutoff in cutoffs:
            for power in powers:
                # Fitting is independent of blend and shrink. Materialize each
                # fitted basis once and create cheap variants.
                horizon_base = _candidate(
                    "horizon_node", fit_point, fit_target, dataset, device,
                    cutoff=cutoff, power=power, ridge=1.0, blend=0.0, shrink=0.0,
                    gate=gate,
                )
                for shrink in shrinks:
                    rows.extend(ratio_variant(horizon_base, blend, shrink) for blend in blends)
                for bin_count in bin_counts:
                    bin_base = _candidate(
                        "pred_bin_horizon", fit_point, fit_target, dataset, device,
                        cutoff=cutoff, power=power, ridge=1.0, blend=0.0, shrink=0.0,
                        bin_count=bin_count, gate=gate,
                    )
                    for shrink in shrinks:
                        rows.extend(ratio_variant(bin_base, blend, shrink) for blend in blends)
    evaluate_ratio_rows(val_point, val_target, rows, dataset, device)
    for row in rows:
        row["score"] = sum(row["val"][key] / baseline[key] for key in METRICS)
        row["eligible"] = all(
            row["val"][key] <= baseline[key] * (1.0 + float(guard_tolerance))
            for key in METRICS
        )
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise RuntimeError("integrated ratio scan produced no guard-eligible identity/candidate")
    if selection == "mape":
        selected = min(eligible, key=lambda row: (row["val"]["mape"], row["val"]["rmse"], row["score"]))
    elif selection == "rmse":
        selected = min(eligible, key=lambda row: (row["val"]["rmse"], row["val"]["mape"], row["score"]))
    else:
        selected = min(eligible, key=lambda row: (row["score"], row["val"]["mape"]))
    def selection_key(row: dict) -> tuple[float, float, float]:
        primary = float(row["score"]) if selection == "score" else float(row["val"][selection])
        return primary, float(row["val"]["mape"]), float(row["score"])

    summary = {
        "selection_protocol": "tocu_history_selected_then_ratio_train_fit_validation_guard_no_test",
        "tail_mode": tail_mode,
        "grid_profile": grid_profile,
        "grid": {
            "bin_counts": list(bin_counts),
            "powers": list(powers),
            "shrinks": list(shrinks),
            "blends": list(blends),
            "cutoff_quantiles": list(cutoff_quantiles),
        },
        "baseline_validation": baseline,
        "candidate_count": len(rows),
        "eligible_count": len(eligible),
        "selected_by": selection,
        "selected": public_row(selected),
        "top_validation_by_selection": [
            public_row(row)
            for row in sorted(
                eligible,
                key=selection_key,
            )[:20]
        ],
    }
    return selected, summary


def blend_history(point: torch.Tensor, full: torch.Tensor, alpha: float) -> torch.Tensor:
    return point + float(alpha) * (full - point)


def resolve_history_cutoff(train: dict, source_metrics: dict, args) -> tuple[str, float | None]:
    """Resolve the history support gate without looking beyond train data.

    ``source`` preserves the released checkpoint's cutoff.  ``train_point_max``
    derives a fresh cutoff from the frozen train point path only, while ``none``
    is retained as an explicit diagnostic mode.  Validation and formal refit
    receive the same resolved value, so the choice cannot drift between stages.
    """
    mode = str(getattr(args, "history_cutoff_mode", "source"))
    if mode not in {"source", "train_point_max", "none"}:
        raise ValueError(f"unsupported history cutoff mode: {mode}")
    if mode == "source":
        raw = source_metrics.get("history_point_cutoff")
        return mode, (float(raw) if raw is not None else None)
    if mode == "train_point_max":
        cutoff = float(train["point"].max().item())
        if not np.isfinite(cutoff):
            raise RuntimeError("train-only history cutoff is not finite")
        return mode, cutoff
    return mode, None


def prepare_history_selection(
    train: dict,
    val: dict,
    feature_median: torch.Tensor,
    feature_scale: torch.Tensor,
    source_metrics: dict,
    args,
    device: torch.device,
):
    train_context_median, train_context_scale = robust_statistics(train["context"])
    train_history_median, train_history_scale = robust_statistics(train["history"])
    train_values = (
        train["point"], train["target"],
        tocu.normalize_features(tocu.replace_point_features(train["raw_features"], train["point"]), feature_median, feature_scale),
        robust_normalize(train["context"], train_context_median, train_context_scale),
        robust_normalize(train["history"], train_history_median, train_history_scale),
        train["times"],
    )
    val_values = (
        val["point"], val["target"],
        tocu.normalize_features(tocu.replace_point_features(val["raw_features"], val["point"]), feature_median, feature_scale),
        robust_normalize(val["context"], train_context_median, train_context_scale),
        robust_normalize(val["history"], train_history_median, train_history_scale),
        val["times"],
    )
    # The history adapter is evaluated over every horizon/node cell.  Keep
    # the cached split tensors on the accelerator so each minibatch does not
    # pay a repeated pageable-CPU transfer.
    train_values_device = tuple(value.to(device) for value in train_values)
    # ``train_history_adapter`` intentionally returns validation predictions
    # on CPU from ``apply_history_adapter``.  Keep the validation tuple on CPU
    # so its blend/reduction contract remains compatible.
    val_values_device = val_values
    cutoff_mode, cutoff = resolve_history_cutoff(train, source_metrics, args)
    formal_args = make_args(args, source_metrics)
    history_overrides = apply_history_overrides(formal_args, args)
    formal_args.dataset = args.dataset
    formal_args.seed = args.seed
    formal_args.batch_size = args.batch_size
    formal_args.grad_clip = args.grad_clip
    formal_args.history_epochs = args.history_selection_epochs
    formal_args.history_patience = args.history_selection_patience
    formal_args.history_selection_metric = args.history_selection
    formal_args.history_blends = tuple(source_metrics.get("history_blends", [0.1, 0.25, 0.5, 0.75, 1.0]))
    formal_args.history_guard_tolerance = float(source_metrics.get("history_guard_tolerance", args.guard_tolerance))
    selected_adapter, selection_history, selected_epoch, selected_mape, selection_baseline, selected_blend, selected_val = train_history_adapter(
        train_values_device[0], train_values_device[1], train_values_device[2], train_values_device[3], train_values_device[4], train_values_device[5],
        val_values_device[0], val_values_device[1], val_values_device[2], val_values_device[3], val_values_device[4], val_values_device[5],
        args.dataset, device, formal_args,
        point_cutoff=cutoff,
    )
    # ``train_history_adapter`` uses epoch 0 and blend 0.0 to represent the
    # identity adapter.  Identity is a valid guarded selection when every
    # non-zero history blend violates at least one metric; the blend grid is
    # intentionally allowed to omit 0.0.  Reject only an actually empty
    # training history.
    if not selection_history:
        raise RuntimeError("integrated history selection produced no guard-eligible epoch")
    train_full = apply_history(selected_adapter, train_values_device, device, args.batch_size, cutoff)
    val_full = apply_history(selected_adapter, val_values_device, device, args.batch_size, cutoff)
    selected_train = blend_history(train["point"], train_full, selected_blend)
    selected_val_point = blend_history(val["point"], val_full, selected_blend)
    recomputed = point_metrics(selected_val_point, val["target"], args.dataset)
    # The adapter helper reports the same selected blend.  A mismatch means
    # the selection and refit paths have silently diverged again.
    for key in METRICS:
        if abs(recomputed[key] - float(selected_val[key])) > 2e-4:
            raise RuntimeError(f"history selection replay mismatch for {key}: {recomputed[key]} vs {selected_val[key]}")
    return {
        "adapter": selected_adapter,
        "selection_history": selection_history,
        "selected_epoch": int(selected_epoch),
        "selected_blend": float(selected_blend),
        "selection_baseline": selection_baseline,
        "selected_val": recomputed,
        "train_values": train_values,
        "val_values": val_values,
        "train_point": selected_train,
        "val_point": selected_val_point,
        "context_median": train_context_median,
        "context_scale": train_context_scale,
        "history_median": train_history_median,
        "history_scale": train_history_scale,
        "cutoff_mode": cutoff_mode,
        "cutoff": cutoff,
        "history_overrides": history_overrides,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--history-selection-epochs", type=int, default=80)
    parser.add_argument("--history-selection-patience", type=int, default=16)
    parser.add_argument("--history-hidden", type=int, default=None)
    parser.add_argument("--history-node-dim", type=int, default=None)
    parser.add_argument("--history-horizon-dim", type=int, default=None)
    parser.add_argument("--history-max-delta", type=float, default=None)
    parser.add_argument("--history-log-weight", type=float, default=None)
    parser.add_argument("--history-relative-weight", type=float, default=None)
    parser.add_argument("--history-relative-target-power", type=float, default=None)
    parser.add_argument("--history-mae-weight", type=float, default=None)
    parser.add_argument("--history-rmse-weight", type=float, default=None)
    parser.add_argument("--history-delta-penalty", type=float, default=None)
    parser.add_argument("--history-horizon-weight", type=float, default=None)
    parser.add_argument("--history-lr", type=float, default=None)
    parser.add_argument("--history-weight-decay", type=float, default=None)
    parser.add_argument("--head-refit-epochs", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--guard-tolerance", type=float, default=0.001)
    parser.add_argument("--history-selection", choices=("score", "rmse", "mape"), default="mape")
    parser.add_argument("--ratio-selection", choices=("score", "rmse", "mape"), default="mape")
    parser.add_argument("--ratio-tail-mode", choices=("low", "upper", "both"), default="low")
    parser.add_argument(
        "--ratio-grid-profile",
        choices=(
            "default",
            "expanded",
            "node_bias",
            "sparse_node_bias",
            "horizon_affine",
            "rmse_tail_affine",
            "rmse_tail_affine_conservative",
            "rmse_tail_affine_ultra_conservative",
            "horizon_affine_gated",
            "pred_bin_affine",
            "horizon_node_affine",
        ),
        default="default",
    )
    parser.add_argument(
        "--history-cutoff-mode",
        choices=("source", "train_point_max", "none"),
        default="source",
        help="History support gate source; train_point_max uses frozen train points only.",
    )
    parser.add_argument(
        "--history-graph",
        action="store_true",
        help="Add causal neighbor-flow summaries to the history adapter features.",
    )
    parser.add_argument(
        "--history-graph-mode",
        choices=("outgoing", "bidirectional", "symmetric"),
        default="outgoing",
        help="Direction policy for optional fixed-graph history summaries.",
    )
    parser.add_argument("--protocol-label", default="TOCU_integrated")
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    source = args.source_run.resolve()
    checkpoint = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=False)
    source_metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    _, model, loaders, scaler, cfg = build_context(args.root.resolve(), args.run_root.resolve(), args.dataset, args.seed, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    capacity = tocu.prepare_capacity(loaders[0], device)
    adjacency = torch.as_tensor(cfg.MODEL.PARAM["adj"], dtype=torch.float32, device=device).abs()
    adjacency.fill_diagonal_(0.0)
    adapter = build_frozen_adapter(model, checkpoint, source_metrics, device)
    splits = [
        collect_frozen_split(
            model, loader, scaler, adjacency, capacity, adapter,
            checkpoint, source_metrics, device,
            include_graph_history=bool(args.history_graph),
            history_graph_mode=args.history_graph_mode,
        )
        for loader in loaders[:2]
    ]
    feature_median = checkpoint["feature_median"]
    feature_scale = checkpoint["feature_scale"]
    for split in splits:
        split["feature_median"] = feature_median
        split["feature_scale"] = feature_scale
    train, val = splits
    history = prepare_history_selection(train, val, feature_median, feature_scale, source_metrics, args, device)
    selected_ratio, ratio_summary = scan_ratio(
        history["train_point"], train["target"], history["val_point"], val["target"],
        args.dataset, device, args.guard_tolerance, args.ratio_selection,
        args.ratio_tail_mode,
        args.ratio_grid_profile,
    )
    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "source_run": str(source),
        "target_sha256_train": sha256_array(train["target"].numpy().astype(np.float64, copy=False)),
        "target_sha256_validation": sha256_array(val["target"].numpy().astype(np.float64, copy=False)),
        "selected_history_epoch": history["selected_epoch"],
        "selected_history_blend": history["selected_blend"],
        "history_cutoff_mode": history["cutoff_mode"],
        "history_support_cutoff": history["cutoff"],
        "history_selection_baseline": history["selection_baseline"],
        "history_selection_val": history["selected_val"],
        "history_selection_metric": args.history_selection,
        "history_overrides": history.get("history_overrides", {}),
        "history_graph_enabled": bool(args.history_graph),
        "history_graph_mode": args.history_graph_mode,
        "ratio": ratio_summary,
        "ratio_tail_mode": args.ratio_tail_mode,
        "ratio_grid_profile": args.ratio_grid_profile,
        "protocol": "TOCU_integrated_history_then_ratio_validation_only_no_test_read",
    }
    destination = args.output.resolve()
    if args.selection_only:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
        return

    # Refit history on train+validation with the selected blend and the
    # train-supported cutoff kept fixed.  This is the same path used in the
    # validation scan, only with more data.
    fit_context_median, fit_context_scale = robust_statistics(torch.cat([train["context"], val["context"]], dim=0))
    fit_history_median, fit_history_scale = robust_statistics(torch.cat([train["history"], val["history"]], dim=0))
    fit_point = torch.cat([train["point"], val["point"]], dim=0)
    fit_target = torch.cat([train["target"], val["target"]], dim=0)
    fit_raw_features = torch.cat([train["raw_features"], val["raw_features"]], dim=0)
    formal_args = make_args(args, source_metrics)
    history_overrides = apply_history_overrides(formal_args, args)
    formal_args.dataset = args.dataset
    formal_args.seed = args.seed
    formal_args.batch_size = args.batch_size
    formal_args.grad_clip = args.grad_clip
    formal_args.history_epochs = history["selected_epoch"]
    formal_args.history_patience = args.history_selection_patience
    fit_values = (
        fit_point,
        fit_target,
        tocu.normalize_features(tocu.replace_point_features(fit_raw_features, fit_point), feature_median, feature_scale),
        robust_normalize(torch.cat([train["context"], val["context"]], dim=0), fit_context_median, fit_context_scale),
        robust_normalize(torch.cat([train["history"], val["history"]], dim=0), fit_history_median, fit_history_scale),
        torch.cat([train["times"], val["times"]], dim=0),
    )
    refit_history, refit_history_log = fit_history_fixed(fit_values, device, formal_args, history["selected_epoch"])
    refit_full = apply_history(refit_history, fit_values, device, args.batch_size, history["cutoff"])
    refit_point = blend_history(fit_point, refit_full, history["selected_blend"])

    # Reconstruct the selected ratio family from its public parameters and fit
    # the values/edges only on the refit train+validation path.
    selected_public = ratio_summary["selected"]
    if selected_public["family"] == "identity":
        ratio_calibration = None
    else:
        ratio_calibration = _candidate(
            selected_public["family"], refit_point, fit_target, args.dataset, device,
            cutoff=selected_public["cutoff"], power=selected_public["power"],
            ridge=selected_public["ridge"], blend=selected_public["blend"],
            shrink=selected_public["shrink"], bin_count=selected_public.get("bin_count"),
            gate=selected_public.get("gate", "below"),
            horizon_start=selected_public.get("horizon_start"),
            bias_cap=selected_public.get("bias_cap"),
            top_k=selected_public.get("top_k"),
        )["calibration"]
    fit_final = tocu.apply_ratio_calibration(refit_point, ratio_calibration)
    fit_head_features = tocu.normalize_features(tocu.replace_point_features(fit_raw_features, fit_final), feature_median, feature_scale)
    _, basis, pc_std, local_std, explained = fit_basis(fit_target - fit_final, int(source_metrics.get("pc_rank", 3)))
    head_epochs = args.head_refit_epochs if args.head_refit_epochs is not None else int(source_metrics.get("head_epochs_completed", 1))
    head, head_history = fit_head_fixed(fit_final, fit_target, fit_head_features, basis, pc_std, local_std, device, formal_args, head_epochs)

    # Held-out labels are touched only here, in the final descriptive call.
    seed_everything(args.seed + 26161)
    tocu.assert_point_path_matches_forward(model, loaders[1], scaler, device)
    totals, means, lowers, uppers, targets, max_dev = tocu.evaluate(
        model, adapter, head, loaders[2], scaler, adjacency, capacity,
        feature_median, feature_scale, checkpoint["mean_directions"], basis,
        pc_std, local_std, args.dataset, device,
        adapter_blend_alpha=float(source_metrics.get("adapter_blend_alpha", 1.0)),
        history_adapter=refit_history,
        history_context_median=fit_context_median,
        history_context_scale=fit_context_scale,
        history_feature_median=fit_history_median,
        history_feature_scale=fit_history_scale,
        history_blend_alpha=history["selected_blend"],
        history_point_cutoff=history["cutoff"],
        ratio_calibration=ratio_calibration,
        history_graph_adjacency=adjacency if bool(args.history_graph) else None,
        history_graph_mode=args.history_graph_mode,
    )
    all_targets = np.concatenate(targets, axis=0)[..., None]
    canonical = np.rint(all_targets).astype(np.float32, copy=False)
    canonical[canonical == 0] = 0.0
    metrics = {
        "MAE": totals["MAE"] / totals["COUNT"],
        "RMSE": (totals["SQ"] / totals["COUNT"]) ** 0.5,
        "MAPE": 100.0 * totals["MAPE"] / max(totals["MAPE_COUNT"], 1.0),
        "CRPS": totals["CRPS_NUM"] / totals["CRPS_DEN"],
        "MIS": totals["MIS"] / totals["COUNT"],
        "model": "TOCU", "model_alias": "TOCU", "dataset": args.dataset, "seed": args.seed,
        "status": "completed", "training_mode": "train_plus_validation_refit_TOCU_integrated",
        "protocol_label": args.protocol_label, "backbone": "frozen_audited_source",
        "adapter_mode": source_metrics.get("adapter_mode", "horizon"),
        "adapter_blend_alpha": source_metrics.get("adapter_blend_alpha", 1.0),
        "selected_backbone_epoch": source_metrics.get("selected_backbone_epoch", 0),
        "selected_adapter_epoch": source_metrics.get("selected_adapter_epoch", 0),
        "selected_history_epoch": history["selected_epoch"], "history_adapter_enabled": True,
        "history_adapter_epochs_completed": history["selected_epoch"],
        "history_blend_alpha": history["selected_blend"], "history_point_cutoff": history["cutoff"],
        "history_cutoff_mode": history["cutoff_mode"],
        "history_context_refit": True, "history_selection_baseline_val": history["selection_baseline"],
        "history_selection_val": history["selected_val"], "history_selection_blend": history["selected_blend"],
        "history_selection_metric": args.history_selection,
        "history_overrides": history_overrides,
        "history_graph_enabled": bool(args.history_graph),
        "history_graph_mode": args.history_graph_mode,
        "ratio_calibration_enabled": ratio_calibration is not None,
        "ratio_calibration_mode": "integrated_refit" if ratio_calibration is not None else "identity",
        "ratio_calibration_selected": public_row(selected_ratio),
        "ratio_calibration_sha256": ratio_hash(ratio_calibration),
        "ratio_selection_protocol": ratio_summary["selection_protocol"],
        "ratio_selection_metric": args.ratio_selection,
        "ratio_tail_mode": args.ratio_tail_mode,
        "ratio_selection_baseline_val": ratio_summary["baseline_validation"],
        "ratio_selection_val": selected_ratio["val"],
        "head_epochs_completed": head_epochs, "selected_head_epoch": head_epochs,
        "pc_rank": int(basis.shape[0]), "pc_explained_variance": explained,
        "mean_direction_count": int(checkpoint["mean_directions"].shape[0]),
        "conditioning_features": list(tocu.FEATURE_NAMES), "max_sample_mean_deviation": max_dev,
        "num_samples": int(source_metrics.get("num_samples", getattr(tocu, "NUM_SAMPLES", 50))),
        "split": [0.6, 0.2, 0.2], "input_len": 12, "output_len": 12,
        "innovation_flags": source_metrics.get("innovation_flags", []),
        "optimization_flags": list(dict.fromkeys(source_metrics.get("optimization_flags", []) + [
            "TOCU_integrated_history_ratio_selection", "history_blend_consistent_refit",
            "history_cutoff_train_supported_refit",
            *( ["causal_graph_history_adapter"] if args.history_graph else [] ),
            *(
                ["causal_graph_history_bidirectional"]
                if args.history_graph and args.history_graph_mode == "bidirectional"
                else []
            ),
            *( ["history_cutoff_train_point_max"] if history["cutoff_mode"] == "train_point_max" else [] ),
            *(
                ["validation_guarded_horizon_node_bias_gate"]
                if selected_public.get("family") == "horizon_node_bias"
                else []
            ),
            *(
                ["validation_guarded_sparse_high_flow_node_bias"]
                if args.ratio_grid_profile == "sparse_node_bias"
                else []
            ),
            *(
                ["validation_guarded_horizon_affine_correction"]
                if selected_public.get("family") == "horizon_affine"
                else []
            ),
            *(
                ["validation_guarded_rmse_tail_affine_correction"]
                if args.ratio_grid_profile in {
                    "rmse_tail_affine",
                    "rmse_tail_affine_conservative",
                    "rmse_tail_affine_ultra_conservative",
                }
                else []
            ),
            *(
                ["validation_guarded_rmse_tail_affine_conservative_correction"]
                if args.ratio_grid_profile == "rmse_tail_affine_conservative"
                else []
            ),
            *(
                ["validation_guarded_rmse_tail_affine_ultra_conservative_correction"]
                if args.ratio_grid_profile == "rmse_tail_affine_ultra_conservative"
                else []
            ),
            *(
                ["validation_guarded_prediction_bin_affine_correction"]
                if selected_public.get("family") == "pred_bin_affine"
                else []
            ),
            *(
                ["validation_guarded_horizon_node_affine_correction"]
                if selected_public.get("family") == "horizon_node_affine"
                else []
            ),
        ])),
        "metric_scales": source_metrics.get("metric_scales", {}),
        "mape_protocol": source_metrics.get("mape_protocol", "target_gt_1"),
        "crps_protocol": source_metrics.get("crps_protocol", "quantile_pinball_q05_to_q95_raw_target_div_sum_abs_target"),
        "mis_interval": source_metrics.get("mis_interval", [0.025, 0.975]),
        "target_sha256": sha256_array(canonical), "target_shape": list(canonical.shape),
        "test_read_policy": "test_loader_consumed_only_by_final_evaluate",
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    source_history = json.loads((source / "history.json").read_text(encoding="utf-8"))
    (output / "history.json").write_text(json.dumps({
        "backbone": source_history.get("backbone", []), "adapter": source_history.get("adapter", []),
        "history_selection": history["selection_history"],
        "history": [{**row, "eligible": True} for row in refit_history_log],
        "head": [{**row, "val_crps": None, "val_nll": None, "val_coverage": None} for row in head_history],
    }, indent=2) + "\n", encoding="utf-8")
    torch.save({
        "model_state_dict": checkpoint["model_state_dict"], "adapter_state_dict": checkpoint["adapter_state_dict"],
        "head_state_dict": head.state_dict(), "feature_median": feature_median, "feature_scale": feature_scale,
        "mean_directions": checkpoint["mean_directions"], "basis": basis, "pc_std": pc_std, "local_std": local_std,
        "adapter_blend_alpha": source_metrics.get("adapter_blend_alpha", 1.0),
        "history_adapter_state_dict": refit_history.state_dict(), "history_context_median": fit_context_median,
        "history_context_scale": fit_context_scale, "history_feature_median": fit_history_median,
        "history_feature_scale": fit_history_scale, "history_blend_alpha": history["selected_blend"],
        "history_point_cutoff": history["cutoff"], "history_cutoff_mode": history["cutoff_mode"],
        "history_graph_enabled": bool(args.history_graph),
        "history_graph_mode": args.history_graph_mode,
        "ratio_calibration": ratio_calibration,
    }, output / "checkpoint.pt")
    np.savez_compressed(output / "summary.npz", mean=np.concatenate(means)[..., None], lower=np.concatenate(lowers)[..., None], upper=np.concatenate(uppers)[..., None], target=np.concatenate(targets, axis=0)[..., None])
    (output / "selection.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output / "DONE").write_text("completed\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
