#!/usr/bin/env python3
"""Train TOCU with balanced point tuning and CRPS-aware uncertainty."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tocu.head import (
    DATASETS,
    NUM_SAMPLES,
    assert_point_path_matches_forward,
    build_context,
    fit_basis,
    hyper_components,
    impedance_risk_from_flow,
    metric_totals,
    seed_everything,
)
from tocu.history import (
    CausalHistoryTemporalAdapter,
    apply_history_adapter,
    apply_history_support_gate,
    causal_context_features,
    collect_causal_features,
    future_time_features,
    history_features,
    robust_normalize,
    robust_statistics,
    train_history_adapter,
)


FEATURE_NAMES = (
    "frequency_energy",
    "impedance_risk",
    "recent_trend",
    "last_flow_mean",
    "point_mean",
    "point_std",
    "max_congestion",
)


def prepare_capacity(loader, device: torch.device) -> torch.Tensor:
    values = []
    for batch in loader:
        raw = torch.as_tensor(batch["inputs"])[..., 0]
        values.append(raw.reshape(-1, raw.shape[-1]))
    return torch.cat(values).quantile(0.95, dim=0).clamp_min(1.0).to(device)


def feature_statistics(train_raw: torch.Tensor):
    median = train_raw.median(0).values
    mad = (train_raw - median).abs().median(0).values
    scale = mad.mul(1.4826).clamp_min(1e-5)
    return median, scale


def normalize_features(raw: torch.Tensor, median: torch.Tensor, scale: torch.Tensor):
    return ((raw - median) / scale).clamp(-5.0, 5.0)


def point_module_features(
    features: torch.Tensor, frequency_conditioned: bool
) -> torch.Tensor:
    """Expose the frequency/impedance inputs only when that module is enabled."""
    if frequency_conditioned:
        return features
    masked = features.clone()
    masked[:, :2] = 0.0
    return masked


def point_adapter_correction(
    adapter,
    features: torch.Tensor,
    directions: torch.Tensor,
    output_shape,
    frequency_conditioned: bool,
):
    """Apply the deterministic point-path part of the frequency module."""
    conditioned = point_module_features(features, frequency_conditioned)
    correction = adapter.correction(conditioned, directions, output_shape)
    if not frequency_conditioned:
        return correction
    frequency_signal = 0.5 * (features[:, 0] + features[:, 1])
    gate = 0.75 + 0.5 * torch.sigmoid(frequency_signal)
    return correction * gate.reshape(-1, *([1] * (correction.ndim - 1)))


def replace_point_features(raw_features: torch.Tensor, point: torch.Tensor):
    updated = raw_features.clone()
    updated[:, 4] = point.mean((1, 2))
    updated[:, 5] = point.std((1, 2), unbiased=False)
    return updated


def forward_with_raw_features(model, raw_inputs, scaler, adjacency, capacity):
    inputs = scaler.transform(raw_inputs.clone())
    point_z, residual_in = hyper_components(model, inputs)
    point = scaler.inverse_transform(point_z)[..., 0]
    flow = raw_inputs[..., 0]
    risk, congestion = impedance_risk_from_flow(flow, capacity, adjacency)
    raw_features = torch.stack(
        [
            residual_in.square().mean((1, 2)),
            risk,
            (flow[:, -1] - flow[:, 0]).mean(-1),
            flow[:, -1].mean(-1),
            point.mean((1, 2)),
            point.std((1, 2), unbiased=False),
            congestion.max(-1).values,
        ],
        dim=1,
    )
    return point, raw_features


def collect_split(model, loader, scaler, device, capacity, adjacency):
    points, targets, raw_features = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            raw_inputs = torch.as_tensor(batch["inputs"], device=device).clone()
            target = torch.as_tensor(batch["target"], device=device)[..., 0]
            point, features = forward_with_raw_features(
                model, raw_inputs, scaler, adjacency, capacity
            )
            points.append(point.cpu())
            targets.append(target.cpu())
            raw_features.append(features.cpu())
    return torch.cat(points), torch.cat(targets), torch.cat(raw_features)


def point_metrics(point: torch.Tensor, target: torch.Tensor, dataset: str):
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


def point_score(current, baseline, rmse_score_weight, mape_score_weight):
    return (
        current["mae"] / baseline["mae"]
        + rmse_score_weight * current["rmse"] / baseline["rmse"]
        + mape_score_weight * current["mape"] / baseline["mape"]
    )


def selection_value(
    current,
    baseline,
    metric,
    rmse_score_weight,
    mape_score_weight,
):
    """Return a validation-only objective while guards protect all point metrics."""
    if metric == "score":
        return point_score(current, baseline, rmse_score_weight, mape_score_weight)
    if metric not in {"mape", "mae", "rmse"}:
        raise ValueError(f"unsupported selection metric: {metric}")
    return current[metric] / baseline[metric]


def center_samples_on_point(samples: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """Keep the independent sample mean equal to the deterministic point forecast."""
    centered = samples.clone()
    sample_count = centered.shape[1]
    centered[:, 0] += sample_count * (point - centered.mean(1))
    return centered


def balanced_point_loss(
    point,
    target,
    dataset,
    rmse_weight,
    relative_weight,
    tail_weight,
    low_flow_weight=0.0,
    horizon_weight=0.0,
    low_flow_power=0.5,
    log_relative_weight=0.0,
    sample_weights=None,
):
    residual = point - target
    threshold = 10.0 if dataset == "Seattle" else 1.0
    element_weights = torch.ones_like(target)
    if horizon_weight:
        horizon = torch.linspace(
            1.0,
            1.0 + float(horizon_weight),
            target.shape[1],
            device=target.device,
            dtype=target.dtype,
        )
        horizon = horizon / horizon.mean().clamp_min(1e-6)
        element_weights = element_weights * horizon.reshape(1, -1, 1)
    if low_flow_weight:
        denominator = target.abs().clamp_min(threshold)
        reference = target.abs().mean().detach().clamp_min(threshold)
        low_factor = (reference / denominator).pow(float(low_flow_power)).clamp(0.5, 3.0)
        element_weights = element_weights * (
            1.0 + float(low_flow_weight) * (low_factor - 1.0).clamp_min(0.0)
        )
    weight_mean = element_weights.mean().clamp_min(1e-6)
    element_weights = element_weights / weight_mean
    scale = target.abs().mean().clamp_min(1.0)
    if sample_weights is None:
        mae = (residual.abs() * element_weights).mean() / scale
        rmse = (residual.square() * element_weights).mean().sqrt() / scale
    else:
        # Keep the historical batch reduction bit-for-bit when no sample
        # profile is requested.  The new path reweights complete windows,
        # which is equivalent to causal hard-example replay without changing
        # the DataLoader order used by the baseline anchor.
        weights = sample_weights.to(device=target.device, dtype=target.dtype)
        weights = weights.clamp_min(1e-4)
        weights = weights / weights.mean().clamp_min(1e-6)
        weighted_elements = element_weights * weights.reshape(-1, 1, 1)
        weighted_denominator = weighted_elements.sum().clamp_min(1e-6)
        mae = (residual.abs() * weighted_elements).sum() / weighted_denominator / scale
        # Use one global weighted RMS.  Per-window sqrt has an undefined
        # zero-point gradient for exact-fit/empty-tail windows and can turn a
        # valid hard-example profile into NaNs after the first optimizer step.
        rmse = (
            (residual.square() * weighted_elements).sum()
            / weighted_denominator
        ).clamp_min(1e-12).sqrt() / scale
    relative_mask = target > threshold
    relative_values = residual.abs() / target.abs().clamp_min(1e-6)
    if sample_weights is None:
        relative = (relative_values * element_weights)[relative_mask].mean()
    else:
        rel_per_sample = (
            (relative_values * element_weights * relative_mask).sum((1, 2))
            / (element_weights * relative_mask).sum((1, 2)).clamp_min(1e-6)
        )
        relative = (rel_per_sample * weights).mean()
    log_residual = (
        torch.log1p(point.clamp_min(0.0))
        - torch.log1p(target.clamp_min(0.0))
    ).abs()
    if sample_weights is None:
        log_relative = (log_residual * element_weights).mean()
    else:
        log_per_sample = (log_residual * element_weights).mean((1, 2))
        log_relative = (log_per_sample * weights).mean()
    tail_threshold = torch.quantile(target.detach().flatten(), 0.75)
    tail_mask = target >= tail_threshold
    if sample_weights is None:
        tail_rmse = (residual.square() * element_weights)[tail_mask].mean().sqrt() / scale
    else:
        tail_elements = weighted_elements * tail_mask
        tail_denominator = tail_elements.sum().clamp_min(1e-6)
        tail_rmse = (
            (residual.square() * tail_elements).sum() / tail_denominator
        ).clamp_min(1e-12).sqrt() / scale
    loss = mae + rmse_weight * rmse
    loss = loss + relative_weight * relative + tail_weight * tail_rmse
    loss = loss + float(log_relative_weight) * log_relative
    return loss, {
        "mae": mae,
        "rmse": rmse,
        "relative": relative,
        "log_relative": log_relative,
        "tail_rmse": tail_rmse,
    }


def build_training_sample_profile(initial_train, dataset, args):
    """Build a train-only window profile for causal hard-example replay.

    The profile is computed from the frozen backbone point and training targets
    only.  It therefore cannot use validation/test labels and remains fixed
    while the backbone is fine-tuned.  Values are normalized to mean one so
    the profile changes which windows receive gradient, not the global loss
    scale.  Zero weights preserve the legacy objective exactly.
    """
    point, target, raw_features = initial_train
    requested = any(
        float(getattr(args, name, 0.0))
        for name in (
            "sample_low_flow_weight",
            "sample_input_low_flow_weight",
            "sample_hard_weight",
            "sample_tail_weight",
        )
    )
    if not requested:
        return None, {"enabled": False}

    threshold = 10.0 if dataset == "Seattle" else 1.0
    target_level = target.abs().mean((1, 2))
    # FEATURE_NAMES[3] is last_flow_mean and is already on the original,
    # causal input scale.  No inverse transform or future target is used.
    input_level = raw_features[:, 3].abs()
    clip = max(float(getattr(args, "sample_weight_clip", 3.0)), 1.0)
    weights = torch.ones_like(target_level, dtype=torch.float32)
    diagnostics = {
        "enabled": True,
        "sample_count": int(target_level.numel()),
        "target_level_median": float(target_level.median()),
        "target_level_q75": float(torch.quantile(target_level, 0.75)),
        "input_level_median": float(input_level.median()),
    }

    low_weight = float(getattr(args, "sample_low_flow_weight", 0.0))
    if low_weight:
        reference = target_level.median().clamp_min(threshold)
        power = float(getattr(args, "sample_low_flow_power", 0.5))
        low_factor = (reference / target_level.clamp_min(threshold)).pow(power)
        low_factor = low_factor.clamp(1.0, clip)
        weights = weights * (1.0 + low_weight * (low_factor - 1.0))
        diagnostics["low_factor_mean"] = float(low_factor.mean())

    input_low_weight = float(getattr(args, "sample_input_low_flow_weight", 0.0))
    if input_low_weight:
        reference = input_level.median().clamp_min(threshold)
        power = float(getattr(args, "sample_input_low_flow_power", 0.5))
        input_factor = (reference / input_level.clamp_min(threshold)).pow(power)
        input_factor = input_factor.clamp(1.0, clip)
        weights = weights * (1.0 + input_low_weight * (input_factor - 1.0))
        diagnostics["input_low_factor_mean"] = float(input_factor.mean())

    hard_weight = float(getattr(args, "sample_hard_weight", 0.0))
    if hard_weight:
        mask = target > threshold
        relative = (point - target).abs() / target.abs().clamp_min(1e-6)
        hard = (relative * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)
        hard_reference = hard.median().clamp_min(1e-4)
        hard_factor = (hard / hard_reference).clamp(0.5, clip)
        weights = weights * (1.0 + hard_weight * (hard_factor - 1.0).clamp_min(0.0))
        diagnostics["hard_relative_median"] = float(hard_reference)
        diagnostics["hard_factor_mean"] = float(hard_factor.mean())

    tail_weight = float(getattr(args, "sample_tail_weight", 0.0))
    if tail_weight:
        q75 = torch.quantile(target_level, 0.75).clamp_min(threshold)
        power = float(getattr(args, "sample_tail_power", 0.5))
        tail_factor = (target_level / q75).pow(power).clamp(1.0, clip)
        weights = weights * (1.0 + tail_weight * (tail_factor - 1.0))
        diagnostics["tail_factor_mean"] = float(tail_factor.mean())

    weights = weights / weights.mean().clamp_min(1e-6)
    weights = weights.clamp(1.0 / clip, clip)
    weights = weights / weights.mean().clamp_min(1e-6)
    diagnostics.update(
        {
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
            "weight_mean": float(weights.mean()),
        }
    )
    return weights.cpu(), diagnostics


def train_backbone(
    model,
    loaders,
    scaler,
    adjacency,
    capacity,
    baseline_train_point,
    dataset,
    device,
    args,
    initial_train=None,
):
    low_flow_weight = getattr(args, "low_flow_weight", 0.0)
    horizon_weight = getattr(args, "horizon_weight", 0.0)
    low_flow_power = getattr(args, "low_flow_power", 0.5)
    if initial_train is None:
        profile_input = (baseline_train_point, None, None)
    else:
        profile_input = initial_train
    if any(
        float(getattr(args, name, 0.0))
        for name in (
            "sample_low_flow_weight",
            "sample_input_low_flow_weight",
            "sample_hard_weight",
            "sample_tail_weight",
        )
    ) and (initial_train is None or initial_train[1] is None or initial_train[2] is None):
        raise ValueError(
            "sample-level training weights require the complete initial_train tuple"
        )
    train_sample_weights, sample_profile = build_training_sample_profile(
        profile_input,
        dataset,
        args,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.backbone_lr, weight_decay=1e-5
    )

    def validation_metrics():
        points, targets = [], []
        for batch in loaders[1]:
            raw_inputs = torch.as_tensor(batch["inputs"], device=device).clone()
            target = torch.as_tensor(batch["target"], device=device)[..., 0]
            point, _ = forward_with_raw_features(
                model, raw_inputs, scaler, adjacency, capacity
            )
            points.append(point.cpu())
            targets.append(target.cpu())
        point = torch.cat(points)
        target = torch.cat(targets)
        full = point_metrics(point, target, dataset)
        tail_fraction = float(getattr(args, "validation_tail_fraction", 0.0))
        if tail_fraction <= 0.0:
            return full, None
        start = int(round((1.0 - tail_fraction) * point.shape[0]))
        start = max(0, min(start, point.shape[0] - 1))
        return full, point_metrics(point[start:], target[start:], dataset)

    model.eval()
    with torch.no_grad():
        baseline, baseline_tail = validation_metrics()
    best_state = copy.deepcopy(model.state_dict())
    selection_metric = getattr(args, "backbone_selection_metric", "score")
    sensitivity_active = bool(getattr(args, "sensitivity_active", False))
    best_epoch = 0
    best_score = point_score(
        baseline, baseline, args.rmse_score_weight, args.mape_score_weight
    )
    best_selection_score = (
        float("inf")
        if sensitivity_active
        else selection_value(
            baseline,
            baseline,
            selection_metric,
            args.rmse_score_weight,
            args.mape_score_weight,
        )
    )
    tail_selection_weight = float(
        getattr(args, "validation_tail_selection_weight", 0.0)
    )
    if not sensitivity_active and baseline_tail is not None and tail_selection_weight:
        best_selection_score += tail_selection_weight * selection_value(
            baseline_tail,
            baseline_tail,
            selection_metric,
            args.rmse_score_weight,
            args.mape_score_weight,
        )
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_values = []
        offset = 0
        progress = (epoch - 1) / max(args.epochs - 1, 1)
        anchor_weight = args.anchor_end + (
            args.anchor_start - args.anchor_end
        ) * (1.0 - progress) ** 2
        for batch in loaders[0]:
            raw_inputs = torch.as_tensor(batch["inputs"], device=device).clone()
            target = torch.as_tensor(batch["target"], device=device)[..., 0]
            point, _ = forward_with_raw_features(
                model, raw_inputs, scaler, adjacency, capacity
            )
            size = point.shape[0]
            sample_weights = None
            if train_sample_weights is not None:
                ramp_power = max(
                    float(getattr(args, "sample_weight_ramp_power", 1.0)),
                    0.0,
                )
                ramp = ((epoch - 1) / max(args.epochs - 1, 1)) ** ramp_power
                sample_weights = 1.0 + ramp * (
                    train_sample_weights[offset : offset + size].to(device)
                    - 1.0
                )
            point_loss, components = balanced_point_loss(
                point,
                target,
                dataset,
                args.rmse_weight,
                args.relative_weight,
                args.tail_weight,
                low_flow_weight,
                horizon_weight,
                low_flow_power,
                args.log_relative_weight,
                sample_weights,
            )
            reference = baseline_train_point[offset : offset + size].to(device)
            scale = target.abs().mean().clamp_min(1.0)
            anchor = F.smooth_l1_loss(point, reference, beta=1.0) / scale
            loss = point_loss + anchor_weight * anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_values.append(
                [
                    float(loss.detach().cpu()),
                    float(components["mae"].detach().cpu()),
                    float(components["rmse"].detach().cpu()),
                    float(components["relative"].detach().cpu()),
                    float(components["log_relative"].detach().cpu()),
                    float(components["tail_rmse"].detach().cpu()),
                    float(anchor.detach().cpu()),
                ]
            )
            offset += size
        model.eval()
        with torch.no_grad():
            val, val_tail = validation_metrics()
        score = point_score(
            val, baseline, args.rmse_score_weight, args.mape_score_weight
        )
        selection_score = selection_value(
            val,
            baseline,
            selection_metric,
            args.rmse_score_weight,
            args.mape_score_weight,
        )
        tail_selection_score = None
        if val_tail is not None and tail_selection_weight:
            tail_selection_score = selection_value(
                val_tail,
                baseline_tail,
                selection_metric,
                args.rmse_score_weight,
                args.mape_score_weight,
            )
            selection_score += tail_selection_weight * tail_selection_score
        eligible = sensitivity_active or all(
            val[name] <= baseline[name] * (1.0 + args.guard_tolerance)
            for name in ("mae", "rmse", "mape")
        )
        train_mean = np.mean(train_values, axis=0)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_mean[0]),
                "train_mae_loss": float(train_mean[1]),
                "train_rmse_loss": float(train_mean[2]),
                "train_relative_loss": float(train_mean[3]),
                "train_log_relative_loss": float(train_mean[4]),
                "train_tail_rmse_loss": float(train_mean[5]),
                "train_anchor_loss": float(train_mean[6]),
                "anchor_weight": anchor_weight,
                "val_mae": val["mae"],
                "val_rmse": val["rmse"],
                "val_mape": val["mape"],
                "val_tail_mae": None if val_tail is None else val_tail["mae"],
                "val_tail_rmse": None if val_tail is None else val_tail["rmse"],
                "val_tail_mape": None if val_tail is None else val_tail["mape"],
                "val_score": score,
                "selection_metric": selection_metric,
                "selection_score": selection_score,
                "tail_selection_score": tail_selection_score,
                "eligible": eligible,
            }
        )
        if eligible and selection_score < best_selection_score - 1e-5:
            best_score = score
            best_selection_score = selection_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    train_backbone.last_sample_profile = sample_profile
    return history, best_score, best_epoch, baseline


class DirectionalMeanAdapter(nn.Module):
    def __init__(self, feature_count: int, direction_scales: torch.Tensor):
        super().__init__()
        self.linear = nn.Linear(feature_count, direction_scales.numel())
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.register_buffer("direction_scales", direction_scales.float())

    def correction(self, features, directions, output_shape):
        coefficients = torch.tanh(self.linear(features)) * self.direction_scales
        flat = coefficients @ directions
        return flat.reshape(features.shape[0], *output_shape)


class HorizonDirectionalAdapter(nn.Module):
    """Directional residual adapter with horizon-conditioned coefficients.

    The external backbone forecast remains the base point.  This adapter only predicts a
    low-rank residual and starts at exactly zero, so it cannot alter the
    checkpoint before validation selects a useful correction.
    """

    def __init__(self, feature_count: int, direction_scales: torch.Tensor, horizon: int):
        super().__init__()
        self.rank = int(direction_scales.numel())
        self.horizon = int(horizon)
        self.linear = nn.Linear(feature_count, self.rank * self.horizon)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.register_buffer("direction_scales", direction_scales.float())

    def correction(self, features, directions, output_shape):
        if len(output_shape) != 2 or output_shape[0] != self.horizon:
            raise ValueError(
                f"horizon adapter expected output shape [B, {self.horizon}, N], "
                f"got [B, {', '.join(map(str, output_shape))}]"
            )
        if directions.shape[0] != self.rank:
            raise ValueError("direction rank does not match adapter rank")
        coefficients = torch.tanh(self.linear(features)).reshape(
            features.shape[0], self.rank, self.horizon
        )
        coefficients = coefficients * self.direction_scales.reshape(1, -1, 1)
        direction_grid = directions.reshape(self.rank, self.horizon, output_shape[1])
        return torch.einsum("bkh,khn->bhn", coefficients, direction_grid)


def adapter_directions(
    errors: torch.Tensor, rank: int, orthogonal_two_source: bool = True
):
    center, basis, pc_std, _, _ = fit_basis(errors, rank)
    center_norm = center.norm().clamp_min(1e-6)
    if orthogonal_two_source:
        # Keep the structural PCs unchanged and move the global residual
        # direction into their orthogonal complement.  This preserves the
        # learned structural coordinates while making the two sources
        # genuinely non-overlapping in the deterministic point path.
        projected = center - torch.einsum("d,rd->r", center, basis) @ basis
        projected_norm = projected.norm()
        if float(projected_norm) > 1e-6:
            center_direction = projected / projected_norm
            center_scale = projected_norm
        else:
            center_direction = center / center_norm
            center_scale = center_norm
        directions = torch.cat([center_direction.unsqueeze(0), basis], dim=0)
        scales = torch.cat([center_scale.reshape(1), pc_std])
    else:
        directions = torch.cat([center.div(center_norm).unsqueeze(0), basis], dim=0)
        scales = torch.cat([center_norm.reshape(1), pc_std])
    return directions.float(), scales.float()


def apply_adapter(
    adapter,
    point,
    features,
    directions,
    device,
    batch_size,
    blend_alpha=1.0,
    frequency_conditioned=True,
):
    outputs = []
    adapter.eval()
    with torch.no_grad():
        for start in range(0, point.shape[0], batch_size):
            stop = min(start + batch_size, point.shape[0])
            correction = point_adapter_correction(
                adapter,
                features[start:stop].to(device),
                directions.to(device),
                point.shape[1:],
                frequency_conditioned,
            )
            outputs.append(point[start:stop] + float(blend_alpha) * correction.cpu())
    return torch.cat(outputs)


def select_adapter_blend(
    adapter,
    val_point,
    val_target,
    val_features,
    directions,
    dataset,
    device,
    args,
):
    """Select a validation-only residual blend under the point-metric guard.

    The adapter is trained for relative/low-flow gains, while this second
    selection keeps its correction from trading away the external backbone's
    MAE or RMSE.  The test split is never inspected here.
    """
    baseline = point_metrics(val_point, val_target, dataset)
    adapted = apply_adapter(
        adapter,
        val_point,
        val_features,
        directions,
        device,
        args.batch_size,
        blend_alpha=1.0,
        frequency_conditioned=not args.disable_frequency_conditioning,
    )
    tolerance_value = getattr(args, "adapter_blend_tolerance", None)
    if tolerance_value is None:
        tolerance_value = args.adapter_guard_tolerance
    tolerance = float(tolerance_value)
    candidates = np.linspace(0.0, 1.0, 21)
    selection_metric = getattr(args, "adapter_blend_selection_metric", "score")
    baseline_selection_score = selection_value(
        baseline,
        baseline,
        selection_metric,
        args.rmse_score_weight,
        args.mape_score_weight,
    )
    best = {
        "alpha": 0.0,
        "metrics": baseline,
        "score": point_score(
            baseline, baseline, args.rmse_score_weight, args.mape_score_weight
        ),
        "selection_score": baseline_selection_score,
        "eligible": True,
    }
    best_selection_score = baseline_selection_score
    rows = []
    for alpha in candidates:
        mixed = val_point + float(alpha) * (adapted - val_point)
        metrics = point_metrics(mixed, val_target, dataset)
        score = point_score(
            metrics, baseline, args.rmse_score_weight, args.mape_score_weight
        )
        selection_score = selection_value(
            metrics,
            baseline,
            selection_metric,
            args.rmse_score_weight,
            args.mape_score_weight,
        )
        eligible = all(
            metrics[name] <= baseline[name] * (1.0 + tolerance)
            for name in ("mae", "rmse", "mape")
        )
        row = {
            "alpha": float(alpha),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mape": metrics["mape"],
            "score": score,
            "selection_metric": selection_metric,
            "selection_score": selection_score,
            "eligible": eligible,
        }
        rows.append(row)
        if eligible and selection_score < best_selection_score - 1e-7:
            best = {
                "alpha": float(alpha),
                "metrics": metrics,
                "score": score,
                "selection_score": selection_score,
                "eligible": True,
            }
            best_selection_score = selection_score
    return best["alpha"], best["metrics"], best["score"], rows


def fit_ratio_calibration(
    prediction,
    target,
    dataset,
    family,
    relative_power,
    cutoff,
    ridge,
    gate="below",
):
    """Fit a train-only multiplicative correction for one forecast regime."""
    if family not in {"global", "horizon", "horizon_node"}:
        raise ValueError(f"unsupported ratio calibration family: {family}")
    if gate not in {"below", "above"}:
        raise ValueError(f"unsupported ratio calibration gate: {gate}")
    threshold = 10.0 if dataset == "Seattle" else 1.0
    weight = 1.0 / target.abs().clamp_min(threshold).pow(float(relative_power))
    if gate == "below":
        supported = prediction < float(cutoff)
    else:
        supported = prediction >= float(cutoff)
    weight = weight * supported.to(weight.dtype)
    numerator = weight * prediction * target
    denominator = weight * prediction.square()
    if family == "horizon":
        numerator = numerator.sum((0, 2))
        denominator = denominator.sum((0, 2))
    elif family == "horizon_node":
        numerator = numerator.sum(0)
        denominator = denominator.sum(0)
    else:
        numerator = numerator.sum()
        denominator = denominator.sum()
    return (numerator + float(ridge)) / (denominator + float(ridge))


def fit_horizon_node_bias(
    prediction,
    target,
    dataset,
    relative_power,
    cutoff,
    ridge,
    gate="above",
    horizon_start=0,
    bias_cap=100.0,
):
    """Fit a train-only additive residual field for a gated tail regime.

    The residual field is estimated per horizon/node, shrunk toward the
    horizon mean with a sample-count-aware ridge, and clipped before it can be
    applied.  This is intentionally separate from multiplicative ratio
    calibration: persistent node-specific under-prediction is additive in the
    original traffic-count scale.
    """
    if gate not in {"below", "above"}:
        raise ValueError(f"unsupported bias calibration gate: {gate}")
    if prediction.ndim != 3 or target.shape != prediction.shape:
        raise ValueError("horizon_node_bias expects prediction/target [batch, horizon, node]")
    threshold = 10.0 if dataset == "Seattle" else 1.0
    target_weight = 1.0 / target.abs().clamp_min(threshold).pow(float(relative_power))
    if gate == "below":
        supported = prediction < float(cutoff)
    else:
        supported = prediction >= float(cutoff)
    horizon_start = int(horizon_start)
    if horizon_start < 0 or horizon_start >= prediction.shape[1]:
        raise ValueError(f"invalid horizon_start={horizon_start}")
    horizon_mask = torch.arange(prediction.shape[1], device=prediction.device).view(1, -1, 1)
    supported = supported & (horizon_mask >= horizon_start)
    weights = target_weight * supported.to(target_weight.dtype)
    residual = target - prediction
    counts = weights.sum(0)
    raw = (weights * residual).sum(0) / counts.clamp_min(1e-6)
    horizon_prior = (weights * residual).sum((0, 2)) / counts.sum(1).clamp_min(1e-6)
    # ridge is relative to the mean supported count, so the public grid is
    # stable across seeds and does not depend on the absolute sample count.
    ridge_mass = float(ridge) * counts.mean(1, keepdim=True).clamp_min(1.0)
    raw = (counts * raw + ridge_mass * horizon_prior[:, None]) / (
        counts + ridge_mass
    ).clamp_min(1e-6)
    raw = torch.where(counts > 0.0, raw, torch.zeros_like(raw))
    return raw.clamp(min=-float(bias_cap), max=float(bias_cap))


def fit_horizon_affine(
    prediction,
    target,
    dataset,
    relative_power,
    ridge,
    horizon_start=0,
    bias_cap=100.0,
    gate="none",
    cutoff=None,
):
    """Fit a train-only horizon affine correction on the original scale.

    The correction is represented as ``slope * prediction + intercept`` and
    is shrunk around the identity by the caller.  Fitting the residual against
    the prediction lets one rule reduce both low-flow over-prediction and
    high-flow under-prediction without using held-out labels.
    """
    if prediction.ndim != 3 or target.shape != prediction.shape:
        raise ValueError("horizon_affine expects prediction/target [batch, horizon, node]")
    if gate not in {"none", "below", "above"}:
        raise ValueError(f"unsupported horizon_affine gate: {gate}")
    if gate != "none" and cutoff is None:
        raise ValueError("gated horizon_affine requires cutoff")
    horizon_start = int(horizon_start)
    if horizon_start < 0 or horizon_start >= prediction.shape[1]:
        raise ValueError(f"invalid horizon_start={horizon_start}")
    threshold = 10.0 if dataset == "Seattle" else 1.0
    weight = 1.0 / target.abs().clamp_min(threshold).pow(float(relative_power))
    weight = weight * (
        torch.arange(prediction.shape[1], device=prediction.device).view(1, -1, 1)
        >= horizon_start
    ).to(weight.dtype)
    if gate == "below":
        weight = weight * (prediction < float(cutoff)).to(weight.dtype)
    elif gate == "above":
        weight = weight * (prediction >= float(cutoff)).to(weight.dtype)
    residual = target - prediction
    sum_w = weight.sum((0, 2))
    sum_x = (weight * prediction).sum((0, 2))
    sum_xx = (weight * prediction.square()).sum((0, 2))
    sum_y = (weight * residual).sum((0, 2))
    sum_xy = (weight * prediction * residual).sum((0, 2))
    ridge_mass = float(ridge) * sum_w.clamp_min(1.0)
    a00 = sum_xx + ridge_mass
    a01 = sum_x
    a11 = sum_w + ridge_mass
    determinant = (a00 * a11 - a01.square()).clamp_min(1e-6)
    slope_delta = (sum_xy * a11 - a01 * sum_y) / determinant
    intercept = (a00 * sum_y - a01 * sum_xy) / determinant
    slope = (1.0 + slope_delta).clamp(min=0.5, max=1.5)
    intercept = intercept.clamp(min=-float(bias_cap), max=float(bias_cap))
    # Keep unsupported early horizons exactly at identity.
    supported = torch.arange(prediction.shape[1], device=prediction.device) >= horizon_start
    slope = torch.where(supported, slope, torch.ones_like(slope))
    intercept = torch.where(supported, intercept, torch.zeros_like(intercept))
    return torch.stack((slope, intercept), dim=1)


def fit_horizon_node_affine(
    prediction,
    target,
    dataset,
    relative_power,
    ridge,
    horizon_start=0,
    bias_cap=100.0,
    gate="none",
    cutoff=None,
):
    """Fit train-only affine maps for each forecast horizon and node."""
    if prediction.ndim != 3 or target.shape != prediction.shape:
        raise ValueError(
            "horizon_node_affine expects prediction/target [batch, horizon, node]"
        )
    if gate not in {"none", "below", "above"}:
        raise ValueError(f"unsupported horizon_node_affine gate: {gate}")
    if gate != "none" and cutoff is None:
        raise ValueError("gated horizon_node_affine requires cutoff")
    horizon_start = int(horizon_start)
    if horizon_start < 0 or horizon_start >= prediction.shape[1]:
        raise ValueError(f"invalid horizon_start={horizon_start}")
    threshold = 10.0 if dataset == "Seattle" else 1.0
    weight = 1.0 / target.abs().clamp_min(threshold).pow(float(relative_power))
    horizon_mask = (
        torch.arange(prediction.shape[1], device=prediction.device).view(1, -1, 1)
        >= horizon_start
    )
    weight = weight * horizon_mask.to(weight.dtype)
    if gate == "below":
        weight = weight * (prediction < float(cutoff)).to(weight.dtype)
    elif gate == "above":
        weight = weight * (prediction >= float(cutoff)).to(weight.dtype)
    sum_w = weight.sum(0)
    sum_x = (weight * prediction).sum(0)
    sum_xx = (weight * prediction.square()).sum(0)
    sum_y = (weight * target).sum(0)
    sum_xy = (weight * prediction * target).sum(0)
    ridge_mass = float(ridge) * sum_w.clamp_min(1.0)
    determinant = (
        (sum_xx + ridge_mass) * (sum_w + ridge_mass) - sum_x.square()
    ).clamp_min(1e-6)
    slope = ((sum_xy + ridge_mass) * (sum_w + ridge_mass) - sum_x * sum_y) / determinant
    intercept = ((sum_xx + ridge_mass) * sum_y - sum_x * (sum_xy + ridge_mass)) / determinant
    slope = torch.where(sum_w > 0.0, slope, torch.ones_like(slope))
    intercept = torch.where(sum_w > 0.0, intercept, torch.zeros_like(intercept))
    slope = slope.clamp(min=0.5, max=1.5)
    intercept = intercept.clamp(min=-float(bias_cap), max=float(bias_cap))
    slope = torch.where(horizon_mask[0], slope, torch.ones_like(slope))
    intercept = torch.where(horizon_mask[0], intercept, torch.zeros_like(intercept))
    return torch.stack((slope, intercept), dim=-1).cpu()


def fit_pred_bin_affine(
    prediction,
    target,
    dataset,
    bin_count,
    relative_power,
    ridge,
    device=None,
    gate="none",
    cutoff=None,
    bias_cap=100.0,
):
    """Fit train-only affine maps for forecast-value bins at each horizon.

    The bin edges and coefficients are derived only from the supplied fit
    tensors.  A ridge prior at ``slope=1, intercept=0`` keeps sparse bins near
    identity, while the public shrink/blend grid controls how much of the map
    reaches validation or the final test path.
    """
    if prediction.ndim != 3 or target.shape != prediction.shape:
        raise ValueError("pred_bin_affine expects prediction/target [batch, horizon, node]")
    bin_count = int(bin_count)
    if bin_count < 2:
        raise ValueError("pred_bin_affine requires at least two bins")
    if gate not in {"none", "below", "above"}:
        raise ValueError(f"unsupported pred_bin_affine gate: {gate}")
    if gate != "none" and cutoff is None:
        raise ValueError("gated pred_bin_affine requires cutoff")
    work_prediction = prediction.to(device=device, dtype=torch.float32)
    work_target = target.to(device=device, dtype=torch.float32)
    threshold = 10.0 if dataset == "Seattle" else 1.0
    quantiles = torch.linspace(0.0, 1.0, bin_count + 1, device=work_prediction.device)
    coefficients = []
    edges = []
    for horizon in range(work_prediction.shape[1]):
        point = work_prediction[:, horizon]
        truth = work_target[:, horizon]
        edge = torch.quantile(point.reshape(-1), quantiles)
        bins = torch.bucketize(point.contiguous(), edge[1:-1], right=True)
        weight = torch.maximum(truth.abs(), torch.as_tensor(threshold, device=truth.device))
        weight = weight.pow(-float(relative_power))
        if gate == "below":
            weight = weight * (point < float(cutoff)).to(weight.dtype)
        elif gate == "above":
            weight = weight * (point >= float(cutoff)).to(weight.dtype)
        rows = []
        for bucket in range(bin_count):
            valid = bins == bucket
            w = weight * valid.to(weight.dtype)
            sum_w = w.sum()
            sum_x = (w * point).sum()
            sum_xx = (w * point.square()).sum()
            sum_y = (w * truth).sum()
            sum_xy = (w * point * truth).sum()
            ridge_mass = float(ridge) * sum_w.clamp_min(1.0)
            determinant = ((sum_xx + ridge_mass) * (sum_w + ridge_mass) - sum_x.square()).clamp_min(1e-6)
            slope = ((sum_xy + ridge_mass) * (sum_w + ridge_mass) - sum_x * sum_y) / determinant
            intercept = ((sum_xx + ridge_mass) * sum_y - sum_x * (sum_xy + ridge_mass)) / determinant
            slope = torch.where(sum_w > 0.0, slope, torch.ones_like(slope))
            intercept = torch.where(sum_w > 0.0, intercept, torch.zeros_like(intercept))
            rows.append(torch.stack((slope.clamp(0.5, 1.5), intercept.clamp(-float(bias_cap), float(bias_cap)))))
        coefficients.append(torch.stack(rows))
        edges.append(edge)
    return torch.stack(coefficients).cpu(), torch.stack(edges).cpu()


def apply_ratio_calibration(point, calibration):
    """Apply a frozen train-fitted ratio rule without touching unsupported flows."""
    if calibration is None:
        return point
    if calibration.get("family") == "pred_bin_horizon":
        edges = calibration["edges"].to(point.device)
        ratio = calibration["ratio"].to(point.device)
        if edges.ndim != 2 or ratio.ndim != 2:
            raise ValueError("pred_bin_horizon calibration expects edges [horizon, bin+1] and ratio [horizon, bin]")
        bin_index = torch.zeros(point.shape, dtype=torch.long, device=point.device)
        for horizon in range(point.shape[1]):
            bin_index[:, horizon] = torch.bucketize(
                point[:, horizon].contiguous(), edges[horizon, 1:-1], right=True
            )
        horizon_index = torch.arange(point.shape[1], device=point.device).view(1, -1, 1)
        factor = ratio[horizon_index, bin_index]
        corrected = point + float(calibration["blend"]) * (point * factor - point)
        gate_mode = calibration.get("gate", "below")
        if gate_mode == "below":
            gate = point < float(calibration["cutoff"])
        elif gate_mode == "above":
            gate = point >= float(calibration["cutoff"])
        else:
            raise ValueError(f"unsupported ratio calibration gate: {gate_mode}")
        return torch.where(gate, corrected.clamp_min(0.0), point)
    if calibration.get("family") == "pred_bin_affine":
        edges = calibration.get("edges")
        ratio = calibration["ratio"].to(point.device)
        if edges is None:
            raise ValueError("pred_bin_affine calibration requires edges")
        edges = edges.to(point.device)
        if edges.ndim != 2 or ratio.ndim != 3 or ratio.shape[-1] != 2:
            raise ValueError(
                "pred_bin_affine calibration expects edges [horizon, bin+1] "
                "and ratio [horizon, bin, 2]"
            )
        bin_index = torch.zeros(point.shape, dtype=torch.long, device=point.device)
        for horizon in range(point.shape[1]):
            bin_index[:, horizon] = torch.bucketize(
                point[:, horizon].contiguous(),
                edges[horizon, 1:-1],
                right=True,
            )
        horizon_index = torch.arange(point.shape[1], device=point.device).view(1, -1, 1)
        slope = ratio[horizon_index, bin_index, 0]
        intercept = ratio[horizon_index, bin_index, 1]
        calibrated = slope * point + intercept
        corrected = point + float(calibration["blend"]) * (calibrated - point)
        gate_mode = calibration.get("gate", "none")
        if gate_mode == "below":
            gate = point < float(calibration["cutoff"])
        elif gate_mode == "above":
            gate = point >= float(calibration["cutoff"])
        elif gate_mode == "none":
            gate = torch.ones_like(point, dtype=torch.bool)
        else:
            raise ValueError(f"unsupported prediction-bin affine gate: {gate_mode}")
        return torch.where(gate, corrected.clamp_min(0.0), point)
    if calibration.get("family") == "horizon_node_affine":
        ratio = calibration["ratio"].to(point.device)
        if ratio.ndim != 3 or ratio.shape[-1] != 2:
            raise ValueError(
                "horizon_node_affine calibration expects ratio [horizon, node, 2]"
            )
        slope = ratio[..., 0].reshape(1, *ratio.shape[:2])
        intercept = ratio[..., 1].reshape(1, *ratio.shape[:2])
        correction = ((slope - 1.0) * point + intercept).clamp(
            min=-float(calibration.get("bias_cap", 100.0)),
            max=float(calibration.get("bias_cap", 100.0)),
        )
        corrected = point + float(calibration["blend"]) * correction
        horizon_start = int(calibration.get("horizon_start", 0))
        horizon_mask = (
            torch.arange(point.shape[1], device=point.device).view(1, -1, 1)
            >= horizon_start
        )
        gate_mode = calibration.get("gate", "none")
        if gate_mode == "below":
            gate = point < float(calibration["cutoff"])
        elif gate_mode == "above":
            gate = point >= float(calibration["cutoff"])
        elif gate_mode == "none":
            gate = torch.ones_like(point, dtype=torch.bool)
        else:
            raise ValueError(f"unsupported horizon-node affine gate: {gate_mode}")
        return torch.where(horizon_mask & gate, corrected.clamp_min(0.0), point)
    ratio = calibration["ratio"].to(point.device)
    family = calibration["family"]
    shrink = float(calibration["shrink"])
    if family == "horizon_node_bias":
        if ratio.ndim != 2:
            raise ValueError("horizon_node_bias bias must have shape [horizon, node]")
        bias = ratio.reshape(1, *ratio.shape)
        corrected = point + float(calibration["blend"]) * bias
        horizon_start = int(calibration.get("horizon_start", 0))
        horizon_mask = torch.arange(point.shape[1], device=point.device).view(1, -1, 1)
        horizon_mask = horizon_mask >= horizon_start
        gate_mode = calibration.get("gate", "above")
        if gate_mode == "below":
            gate = point < float(calibration["cutoff"])
        elif gate_mode == "above":
            gate = point >= float(calibration["cutoff"])
        else:
            raise ValueError(f"unsupported ratio calibration gate: {gate_mode}")
        return torch.where(
            gate & horizon_mask,
            corrected.clamp_min(0.0),
            point,
        )
    if family == "horizon_affine":
        if ratio.ndim != 2 or ratio.shape[1] != 2:
            raise ValueError("horizon_affine correction must have shape [horizon, 2]")
        slope = ratio[:, 0].reshape(1, -1, 1)
        intercept = ratio[:, 1].reshape(1, -1, 1)
        correction = ((slope - 1.0) * point + intercept).clamp(
            min=-float(calibration.get("bias_cap", 100.0)),
            max=float(calibration.get("bias_cap", 100.0)),
        )
        corrected = point + float(calibration["blend"]) * correction
        horizon_start = int(calibration.get("horizon_start", 0))
        horizon_mask = torch.arange(point.shape[1], device=point.device).view(1, -1, 1) >= horizon_start
        gate_mode = calibration.get("gate", "none")
        if gate_mode == "below":
            gate = point < float(calibration["cutoff"])
        elif gate_mode == "above":
            gate = point >= float(calibration["cutoff"])
        elif gate_mode == "none":
            gate = torch.ones_like(point, dtype=torch.bool)
        else:
            raise ValueError(f"unsupported horizon affine gate: {gate_mode}")
        return torch.where(horizon_mask & gate, corrected.clamp_min(0.0), point)
    if family == "horizon":
        factor = 1.0 + (1.0 - shrink) * (ratio.reshape(1, -1, 1) - 1.0)
    elif family == "horizon_node":
        if ratio.ndim != 2:
            raise ValueError("horizon_node ratio must have shape [horizon, node]")
        factor = 1.0 + (1.0 - shrink) * (ratio.reshape(1, *ratio.shape) - 1.0)
    else:
        factor = 1.0 + (1.0 - shrink) * (ratio - 1.0)
    corrected = point + float(calibration["blend"]) * (point * factor - point)
    gate_mode = calibration.get("gate", "below")
    if gate_mode == "below":
        gate = point < float(calibration["cutoff"])
    elif gate_mode == "above":
        gate = point >= float(calibration["cutoff"])
    else:
        raise ValueError(f"unsupported ratio calibration gate: {gate_mode}")
    return torch.where(gate, corrected.clamp_min(0.0), point)


def ratio_payload_sha256(calibration):
    """Hash ratio metadata and values so replay cannot silently use another rule."""
    if calibration is None:
        return None
    ratio = calibration.get("ratio")
    if ratio is None:
        raise ValueError("ratio calibration payload has no ratio tensor")
    metadata = {
        key: value
        for key, value in calibration.items()
        if key not in {"ratio", "edges"}
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str).encode()
    values = np.ascontiguousarray(ratio.detach().cpu().numpy())
    digest = hashlib.sha256()
    digest.update(encoded)
    digest.update(str(values.dtype).encode())
    digest.update(json.dumps(list(values.shape), separators=(",", ":")).encode())
    digest.update(values.tobytes())
    edges = calibration.get("edges")
    if edges is not None:
        edge_values = np.ascontiguousarray(edges.detach().cpu().numpy())
        digest.update(b"edges")
        digest.update(str(edge_values.dtype).encode())
        digest.update(json.dumps(list(edge_values.shape), separators=(",", ":")).encode())
        digest.update(edge_values.tobytes())
    return digest.hexdigest()


def select_ratio_calibration(
    train_point,
    train_target,
    val_point,
    val_target,
    dataset,
    args,
):
    """Select a low-flow ratio rule using train fitting and validation guards only."""
    mode = getattr(args, "ratio_calibration_mode", "off")
    if mode == "off":
        return None, None, []
    threshold = 10.0 if dataset == "Seattle" else 1.0
    values = train_point[train_target > threshold].reshape(-1)
    if values.numel() > 2_000_000:
        stride = max(1, values.numel() // 2_000_000)
        values = values[::stride][:2_000_000]
    quantile_levels = (0.70, 0.85, 0.95, 1.0)
    cutoffs = [float(torch.quantile(values, torch.tensor(q))) for q in quantile_levels]
    baseline = point_metrics(val_point, val_target, dataset)
    rows = []
    families = ("horizon",) if mode == "horizon" else ("global", "horizon")
    for family in families:
        for cutoff in cutoffs:
            for relative_power in (0.5, 1.0):
                ratio = fit_ratio_calibration(
                    train_point,
                    train_target,
                    dataset,
                    family,
                    relative_power,
                    cutoff,
                    1.0,
                )
                for blend in (0.25, 0.50):
                    for shrink in (0.0, 0.5):
                        candidate = {
                            "family": family,
                            "cutoff": cutoff,
                            "relative_power": relative_power,
                            "ridge": 1.0,
                            "blend": blend,
                            "shrink": shrink,
                            "ratio": ratio.detach().cpu(),
                        }
                        val = point_metrics(
                            apply_ratio_calibration(val_point, candidate),
                            val_target,
                            dataset,
                        )
                        candidate.update(
                            {
                                "val_mae": val["mae"],
                                "val_rmse": val["rmse"],
                                "val_mape": val["mape"],
                                "eligible": all(
                                    val[name]
                                    <= baseline[name]
                                    * (1.0 + float(args.ratio_guard_tolerance))
                                    for name in ("mae", "rmse", "mape")
                                ),
                            }
                        )
                        rows.append(candidate)
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return None, baseline, rows
    selected = min(eligible, key=lambda row: (row["val_mape"], row["val_rmse"]))
    return selected, baseline, rows


def train_adapter(
    train_point,
    train_target,
    train_features,
    val_point,
    val_target,
    val_features,
    directions,
    direction_scales,
    dataset,
    device,
    args,
):
    low_flow_weight = getattr(args, "adapter_low_flow_weight", None)
    if low_flow_weight is None:
        low_flow_weight = getattr(args, "low_flow_weight", 0.0)
    horizon_weight = getattr(args, "adapter_horizon_weight", None)
    if horizon_weight is None:
        horizon_weight = getattr(args, "horizon_weight", 0.0)
    low_flow_power = getattr(args, "adapter_low_flow_power", None)
    if low_flow_power is None:
        low_flow_power = getattr(args, "low_flow_power", 0.5)
    log_relative_weight = getattr(args, "adapter_log_relative_weight", None)
    if log_relative_weight is None:
        log_relative_weight = getattr(args, "log_relative_weight", 0.0)
    if getattr(args, "adapter_mode", "directional") == "horizon":
        adapter = HorizonDirectionalAdapter(
            train_features.shape[1], direction_scales.to(device), train_point.shape[1]
        ).to(device)
    else:
        adapter = DirectionalMeanAdapter(
            train_features.shape[1], direction_scales.to(device)
        ).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.adapter_lr, weight_decay=1e-4
    )
    baseline = point_metrics(val_point, val_target, dataset)
    best_state = copy.deepcopy(adapter.state_dict())
    best_score = point_score(
        baseline, baseline, args.rmse_score_weight, args.mape_score_weight
    )
    selection_metric = getattr(args, "adapter_selection_metric", "score")
    if selection_metric not in {"score", "mape", "mae", "rmse"}:
        raise ValueError(f"unsupported adapter selection metric: {selection_metric}")
    best_selection_score = 1.0 if selection_metric != "score" else best_score
    best_epoch = 0
    stale = 0
    history = []
    directions_device = directions.to(device)
    for epoch in range(1, args.adapter_epochs + 1):
        adapter.train()
        order = torch.randperm(train_point.shape[0])
        losses = []
        for start in range(0, order.numel(), args.batch_size):
            index = order[start : start + args.batch_size]
            point = train_point[index].to(device)
            target = train_target[index].to(device)
            features = train_features[index].to(device)
            correction = point_adapter_correction(
                adapter,
                features,
                directions_device,
                point.shape[1:],
                not args.disable_frequency_conditioning,
            )
            corrected = point + correction
            loss, _ = balanced_point_loss(
                corrected,
                target,
                dataset,
                args.rmse_weight,
                args.relative_weight,
                args.tail_weight,
                low_flow_weight,
                horizon_weight,
                low_flow_power,
                log_relative_weight,
            )
            scale = target.abs().mean().clamp_min(1.0)
            penalty = correction.abs().mean() / scale
            loss = loss + args.adapter_penalty * penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        corrected_val = apply_adapter(
            adapter,
            val_point,
            val_features,
            directions,
            device,
            args.batch_size,
            frequency_conditioned=not args.disable_frequency_conditioning,
        )
        val = point_metrics(corrected_val, val_target, dataset)
        score = point_score(
            val, baseline, args.rmse_score_weight, args.mape_score_weight
        )
        if selection_metric == "mape":
            selection_score = val["mape"] / baseline["mape"]
        elif selection_metric == "mae":
            selection_score = val["mae"] / baseline["mae"]
        elif selection_metric == "rmse":
            selection_score = val["rmse"] / baseline["rmse"]
        else:
            selection_score = score
        eligible = all(
            val[name] <= baseline[name] * (1.0 + args.adapter_guard_tolerance)
            for name in ("mae", "rmse", "mape")
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_mae": val["mae"],
                "val_rmse": val["rmse"],
                "val_mape": val["mape"],
                "val_score": score,
                "selection_metric": selection_metric,
                "selection_score": selection_score,
                "eligible": eligible,
            }
        )
        if eligible and selection_score < best_selection_score - 1e-5:
            best_state = copy.deepcopy(adapter.state_dict())
            best_score = score
            best_selection_score = selection_score
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= args.adapter_patience:
                break
    adapter.load_state_dict(best_state)
    return adapter, history, best_score, best_epoch, baseline


class CRPSUncertaintyHead(nn.Module):
    def __init__(self, feature_count, struct_var, local_var, frequency_conditioned=True):
        super().__init__()
        self.register_buffer("struct_var", struct_var.float())
        self.register_buffer("local_var", local_var.float())
        self.frequency_conditioned = bool(frequency_conditioned)
        self.gate = nn.Sequential(
            nn.Linear(feature_count, 24),
            nn.Tanh(),
            nn.Linear(24, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.log_struct_scale = nn.Parameter(torch.tensor(-0.25))
        self.log_local_scale = nn.Parameter(torch.tensor(0.0))

    def components(self, features):
        if self.frequency_conditioned:
            # Keep the declared frequency/impedance conditioning active even
            # when the learned projection is initialized at zero or learns a
            # near-constant correction on a particular dataset.
            frequency_signal = 0.5 * (features[:, 0] + features[:, 1])
            gate_logits = self.gate(features).squeeze(-1) + 0.25 * frequency_signal
            gate = 0.75 + 0.5 * torch.sigmoid(gate_logits)
        else:
            gate = torch.ones(features.shape[0], device=features.device, dtype=features.dtype)
        struct_scale = self.log_struct_scale.exp().clamp(1e-3, 10.0)
        local_scale = self.log_local_scale.exp().clamp(1e-3, 10.0)
        return gate, struct_scale, local_scale

    def std(self, features):
        gate, struct_scale, local_scale = self.components(features)
        variance = (
            (struct_scale * gate[:, None]).square() * self.struct_var[None]
            + local_scale.square() * self.local_var[None]
        )
        return variance.clamp_min(1e-6).sqrt()


def gaussian_crps(mean, std, target):
    std = std.clamp_min(1e-4)
    z = (target - mean) / std
    phi = torch.exp(-0.5 * z.square()) / float((2.0 * np.pi) ** 0.5)
    cdf = 0.5 * (1.0 + torch.erf(z / float(2.0**0.5)))
    return std * (
        z * (2.0 * cdf - 1.0) + 2.0 * phi - float(np.pi ** -0.5)
    )


def gaussian_metric_totals(
    mean: torch.Tensor,
    std: torch.Tensor,
    target: torch.Tensor,
    dataset: str,
) -> dict[str, float]:
    """Compute point, interval, and analytic Gaussian CRPS totals."""
    std = std.clamp_min(1e-4)
    lower = mean - 1.96 * std
    upper = mean + 1.96 * std
    interval = (upper - lower) + 40.0 * (
        (lower - target).clamp_min(0.0) + (target - upper).clamp_min(0.0)
    )
    error = (mean - target).abs()
    residual = mean - target
    mask = target > (10.0 if dataset == "Seattle" else 1.0)
    crps = gaussian_crps(mean, std, target)
    return {
        "MAE": float(error.sum()),
        "SQ": float(residual.square().sum()),
        "COUNT": float(target.numel()),
        "MAPE": float(
            (error / target.abs().clamp_min(1e-5)).masked_select(mask).sum()
        ),
        "MAPE_COUNT": float(mask.sum()),
        "CRPS_NUM": float(crps.sum()),
        "CRPS_DEN": float(target.abs().sum()),
        "MIS": float(interval.sum()),
    }


def projected_local_variance(
    local_std: torch.Tensor,
    basis: torch.Tensor,
    orthogonal_two_source: bool,
) -> torch.Tensor:
    """Return the exact diagonal variance after the local-source projection."""
    local_var = local_std.square()
    if not orthogonal_two_source:
        return local_var
    dimension = basis.shape[1]
    projector = torch.eye(dimension, dtype=basis.dtype, device=basis.device)
    projector = projector - basis.T @ basis
    # z_local is row-oriented, so diag(P diag(v) P) is the column-wise sum.
    return (projector.square() * local_var.reshape(-1, 1)).sum(0).clamp_min(1e-8)


def train_uncertainty_head(
    train_point,
    train_target,
    train_features,
    val_point,
    val_target,
    val_features,
    basis,
    pc_std,
    local_std,
    device,
    args,
    orthogonal_two_source=True,
    analytic_probabilistic_eval=False,
):
    struct_var = (basis.T * pc_std.reshape(1, -1)).square().sum(1)
    local_var = (
        projected_local_variance(
            local_std,
            basis,
            orthogonal_two_source=orthogonal_two_source,
        )
        if analytic_probabilistic_eval
        else local_std.square()
    )
    head = CRPSUncertaintyHead(
        train_features.shape[1], struct_var.to(device), local_var.to(device),
        frequency_conditioned=not args.disable_frequency_conditioning,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.head_lr, weight_decay=1e-4
    )

    def validation():
        nll_sum = crps_num = crps_den = covered_sum = 0.0
        count = 0
        head.eval()
        with torch.no_grad():
            for start in range(0, val_point.shape[0], args.batch_size):
                stop = min(start + args.batch_size, val_point.shape[0])
                point = val_point[start:stop].reshape(stop - start, -1).to(device)
                target = val_target[start:stop].reshape(stop - start, -1).to(device)
                features = val_features[start:stop].to(device)
                std = head.std(features)
                residual = target - point
                nll_value = 0.5 * ((residual / std).square() + 2.0 * std.log())
                crps_value = gaussian_crps(point, std, target)
                covered = (target >= point - 1.96 * std) & (target <= point + 1.96 * std)
                nll_sum += float(nll_value.sum().cpu())
                crps_num += float(crps_value.sum().cpu())
                crps_den += float(target.abs().sum().clamp_min(1.0).cpu())
                covered_sum += float(covered.sum().cpu())
                count += target.numel()
        nll_value = nll_sum / count
        crps_value = crps_num / crps_den
        coverage = covered_sum / count
        score = (
            args.head_nll_weight * nll_value
            + args.head_crps_weight * crps_value
            + args.coverage_weight * abs(coverage - 0.95)
        )
        return {
            "nll": nll_value,
            "crps": crps_value,
            "coverage": coverage,
            "score": score,
            "count": count,
        }

    best_state = copy.deepcopy(head.state_dict())
    baseline = validation()
    best_score = baseline["crps"]
    best_epoch = 0
    stale = 0
    history = []
    eligible_epoch_count = 0
    best_fallback_state = None
    best_fallback_key = None
    best_fallback_epoch = 0
    best_any_state = None
    best_any_key = None
    best_any_epoch = 0
    for epoch in range(1, args.head_epochs + 1):
        head.train()
        order = torch.randperm(train_point.shape[0])
        losses = []
        for start in range(0, order.numel(), args.batch_size):
            index = order[start : start + args.batch_size]
            point = train_point[index].reshape(index.numel(), -1).to(device)
            target = train_target[index].reshape(index.numel(), -1).to(device)
            features = train_features[index].to(device)
            std = head.std(features)
            residual = target - point
            nll = 0.5 * ((residual / std).square() + 2.0 * std.log()).mean()
            scale = target.abs().mean().clamp_min(1.0)
            crps = gaussian_crps(point, std, target).mean() / scale
            loss = args.head_nll_weight * nll + args.head_crps_weight * crps
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = validation()
        coverage_floor = max(
            0.85,
            baseline["coverage"] - args.head_coverage_tolerance,
        )
        coverage_ok = val["coverage"] >= coverage_floor
        nll_ok = val["nll"] <= baseline["nll"] * (1.0 + args.head_nll_tolerance)
        eligible = nll_ok and coverage_ok
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_nll": val["nll"],
                "val_crps": val["crps"],
                "val_coverage": val["coverage"],
                "val_score": val["score"],
                "coverage_ok": coverage_ok,
                "nll_ok": nll_ok,
                "eligible": eligible,
            }
        )
        if eligible:
            eligible_epoch_count += 1
        fallback_key = (val["crps"], val["score"], epoch)
        if coverage_ok and (
            best_fallback_key is None or fallback_key < best_fallback_key
        ):
            best_fallback_state = copy.deepcopy(head.state_dict())
            best_fallback_key = fallback_key
            best_fallback_epoch = epoch
        if best_any_key is None or (val["score"], epoch) < best_any_key:
            best_any_state = copy.deepcopy(head.state_dict())
            best_any_key = (val["score"], epoch)
            best_any_epoch = epoch
        eligible = history[-1]["eligible"]
        if eligible and val["crps"] < best_score - 1e-6:
            best_state = copy.deepcopy(head.state_dict())
            best_score = val["crps"]
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= args.head_patience:
                break
    if eligible_epoch_count == 0:
        if best_fallback_state is not None:
            best_state = best_fallback_state
            best_score = best_fallback_key[0]
            best_epoch = best_fallback_epoch
            selection_mode = "coverage_fallback"
        elif best_any_state is not None:
            best_state = best_any_state
            best_score = best_any_key[0]
            best_epoch = best_any_epoch
            selection_mode = "objective_fallback"
        else:
            selection_mode = "initial_state"
    elif best_epoch == 0:
        selection_mode = "guarded_baseline"
    else:
        selection_mode = "guarded_crps"
    head.load_state_dict(best_state)
    return (
        head,
        history,
        best_score,
        best_epoch,
        baseline,
        selection_mode,
        eligible_epoch_count,
    )


def evaluate(
    model,
    adapter,
    head,
    loader,
    scaler,
    adjacency,
    capacity,
    feature_median,
    feature_scale,
    adapter_directions_value,
    basis,
    pc_std,
    local_std,
    dataset,
    device,
    reference_model=None,
    blend_alpha=1.0,
    adapter_blend_alpha=1.0,
    history_adapter=None,
    history_context_median=None,
    history_context_scale=None,
    history_feature_median=None,
    history_feature_scale=None,
    history_blend_alpha=0.0,
    history_point_cutoff=None,
    ratio_calibration=None,
    history_graph_adjacency=None,
    history_graph_mode="outgoing",
    orthogonal_two_source=True,
    mean_consistency=True,
    frequency_conditioned=True,
    analytic_probabilistic_eval=False,
):
    totals = {
        key: 0.0
        for key in (
            "MAE",
            "SQ",
            "COUNT",
            "MAPE",
            "MAPE_COUNT",
            "CRPS_NUM",
            "CRPS_DEN",
            "MIS",
        )
    }
    means, stds, lowers, uppers, targets = [], [], [], [], []
    max_mean_deviation = 0.0
    model.eval()
    if reference_model is not None:
        reference_model.eval()
    adapter.eval()
    head.eval()
    if history_adapter is not None:
        history_adapter.eval()
        if any(
            value is None
            for value in (
                history_context_median,
                history_context_scale,
                history_feature_median,
                history_feature_scale,
            )
        ):
            raise ValueError("history adapter requires all causal feature statistics")
    struct_direction = basis * pc_std.reshape(-1, 1)
    with torch.no_grad():
        for batch in loader:
            raw_inputs = torch.as_tensor(batch["inputs"], device=device).clone()
            target = torch.as_tensor(batch["target"], device=device)[..., 0]
            point, raw_features = forward_with_raw_features(
                model, raw_inputs, scaler, adjacency, capacity
            )
            if reference_model is not None:
                reference_point, _ = forward_with_raw_features(
                    reference_model, raw_inputs, scaler, adjacency, capacity
                )
                point = (
                    (1.0 - blend_alpha) * reference_point
                    + blend_alpha * point
                )
                raw_features = replace_point_features(raw_features, point)
            features = normalize_features(
                raw_features, feature_median.to(device), feature_scale.to(device)
            )
            correction = point_adapter_correction(
                adapter,
                features,
                adapter_directions_value.to(device),
                point.shape[1:],
                frequency_conditioned,
            )
            point = point + float(adapter_blend_alpha) * correction
            raw_features = replace_point_features(raw_features, point)
            if history_adapter is not None and float(history_blend_alpha) > 0.0:
                flow = raw_inputs[..., 0]
                _, congestion = impedance_risk_from_flow(
                    flow, capacity, adjacency
                )
                context = causal_context_features(flow, congestion)
                history = history_features(
                    flow, history_graph_adjacency, graph_mode=history_graph_mode
                )
                future_time = future_time_features(
                    raw_inputs, model.daily_len, model.pred_len
                )
                context = robust_normalize(
                    context,
                    history_context_median.to(device),
                    history_context_scale.to(device),
                )
                history = robust_normalize(
                    history,
                    history_feature_median.to(device),
                    history_feature_scale.to(device),
                )
                corrected = history_adapter(
                    point,
                    point_module_features(features, frequency_conditioned),
                    context,
                    history,
                    future_time,
                )[0]
                corrected = apply_history_support_gate(
                    point, corrected, history_point_cutoff
                )
                point = point + float(history_blend_alpha) * (corrected - point)
                raw_features = replace_point_features(raw_features, point)
            point = apply_ratio_calibration(point, ratio_calibration)
            raw_features = replace_point_features(raw_features, point)
            features = normalize_features(
                raw_features, feature_median.to(device), feature_scale.to(device)
            )
            if analytic_probabilistic_eval:
                flat_point = point.reshape(point.shape[0], -1)
                flat_target = target.reshape(target.shape[0], -1)
                std = head.std(features)
                mean_cpu = flat_point.cpu()
                std_cpu = std.cpu()
                target_cpu = flat_target.cpu()
                batch_totals = gaussian_metric_totals(
                    mean_cpu,
                    std_cpu,
                    target_cpu,
                    dataset,
                )
                for key in totals:
                    totals[key] += batch_totals[key]
                means.append(mean_cpu.reshape(point.shape).numpy().astype(np.float32))
                stds.append(std_cpu.reshape(point.shape).numpy().astype(np.float32))
                lowers.append(
                    (mean_cpu - 1.96 * std_cpu)
                    .reshape(point.shape)
                    .numpy()
                    .astype(np.float32)
                )
                uppers.append(
                    (mean_cpu + 1.96 * std_cpu)
                    .reshape(point.shape)
                    .numpy()
                    .astype(np.float32)
                )
                targets.append(target_cpu.reshape(point.shape).numpy().astype(np.float32))
                # The analytic mean is the deterministic point forecast exactly;
                # no finite-sample centering correction is applied.
                continue
            gate, struct_scale, local_scale = head.components(features)
            flat_point = point.reshape(point.shape[0], -1)
            z_pc = torch.randn(flat_point.shape[0], NUM_SAMPLES, basis.shape[0])
            structural = torch.einsum("bsk,kl->bsl", z_pc, struct_direction)
            z_local = torch.randn(flat_point.shape[0], NUM_SAMPLES, basis.shape[1])
            z_local = z_local * local_std.reshape(1, 1, -1)
            if orthogonal_two_source:
                z_local = z_local - torch.einsum(
                    "bsk,kl->bsl",
                    torch.einsum("bsl,kl->bsk", z_local, basis),
                    basis,
                )
            noise = (
                struct_scale.cpu() * gate[:, None, None].cpu() * structural
                + local_scale.cpu() * z_local
            )
            samples = (flat_point.cpu()[:, None] + noise).reshape(
                point.shape[0], NUM_SAMPLES, *point.shape[1:]
            )
            if mean_consistency:
                samples = center_samples_on_point(samples, point.cpu())
            max_mean_deviation = max(
                max_mean_deviation,
                float((samples.mean(1) - point.cpu()).abs().max()),
            )
            batch_totals = metric_totals(samples, target.cpu(), dataset)
            for key in totals:
                totals[key] += batch_totals[key]
            means.append(samples.mean(1).numpy().astype(np.float32))
            lowers.append(samples.quantile(0.025, dim=1).numpy().astype(np.float32))
            uppers.append(samples.quantile(0.975, dim=1).numpy().astype(np.float32))
            targets.append(target.cpu().numpy().astype(np.float32))
    if analytic_probabilistic_eval:
        return totals, means, stds, lowers, uppers, targets, max_mean_deviation
    return totals, means, lowers, uppers, targets, max_mean_deviation


def run(args):
    seed_everything(args.seed)
    import time
    started_at = time.perf_counter()
    root = Path(args.root).resolve()
    run_root = Path(args.run_root).resolve()
    output = Path(args.output).resolve()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    effective_adapter_low_flow_weight = (
        args.adapter_low_flow_weight
        if args.adapter_low_flow_weight is not None
        else args.low_flow_weight
    )
    effective_adapter_horizon_weight = (
        args.adapter_horizon_weight
        if args.adapter_horizon_weight is not None
        else args.horizon_weight
    )
    effective_adapter_low_flow_power = (
        args.adapter_low_flow_power
        if args.adapter_low_flow_power is not None
        else args.low_flow_power
    )
    effective_adapter_log_relative_weight = (
        args.adapter_log_relative_weight
        if args.adapter_log_relative_weight is not None
        else args.log_relative_weight
    )
    _, model, loaders, scaler, cfg = build_context(
        root, run_root, args.dataset, args.seed, device
    )
    capacity = prepare_capacity(loaders[0], device)
    adjacency = torch.as_tensor(
        cfg.MODEL.PARAM["adj"], dtype=torch.float32, device=device
    ).abs()
    adjacency.fill_diagonal_(0.0)

    initial_train = collect_split(
        model, loaders[0], scaler, device, capacity, adjacency
    )
    backbone_history, best_point_score, selected_backbone_epoch, baseline_val = (
        train_backbone(
            model,
            loaders,
            scaler,
            adjacency,
            capacity,
            initial_train[0],
            args.dataset,
            device,
            args,
            initial_train=initial_train,
        )
    )
    tuned_train = collect_split(
        model, loaders[0], scaler, device, capacity, adjacency
    )
    tuned_val = collect_split(
        model, loaders[1], scaler, device, capacity, adjacency
    )
    feature_median, feature_scale = feature_statistics(tuned_train[2])
    train_features = normalize_features(
        tuned_train[2], feature_median, feature_scale
    )
    val_features = normalize_features(tuned_val[2], feature_median, feature_scale)
    frequency_conditioned = not args.disable_frequency_conditioning
    point_train_features = point_module_features(train_features, frequency_conditioned)
    point_val_features = point_module_features(val_features, frequency_conditioned)

    mean_directions, mean_direction_scales = adapter_directions(
        tuned_train[1] - tuned_train[0],
        args.pc_rank,
        orthogonal_two_source=not args.disable_orthogonal_two_source,
    )
    adapter, adapter_history, best_adapter_score, selected_adapter_epoch, adapter_baseline = (
        train_adapter(
            tuned_train[0],
            tuned_train[1],
            train_features,
            tuned_val[0],
            tuned_val[1],
            val_features,
            mean_directions,
            mean_direction_scales,
            args.dataset,
            device,
            args,
        )
    )
    (
        adapter_blend_alpha,
        adapter_blend_val,
        adapter_blend_score,
        adapter_blend_rows,
    ) = select_adapter_blend(
        adapter,
        tuned_val[0],
        tuned_val[1],
        val_features,
        mean_directions,
        args.dataset,
        device,
        args,
    )
    adapter_corrected_train = apply_adapter(
        adapter,
        tuned_train[0],
        train_features,
        mean_directions,
        device,
        args.batch_size,
        blend_alpha=adapter_blend_alpha,
        frequency_conditioned=not args.disable_frequency_conditioning,
    )
    adapter_corrected_val = apply_adapter(
        adapter,
        tuned_val[0],
        val_features,
        mean_directions,
        device,
        args.batch_size,
        blend_alpha=adapter_blend_alpha,
        frequency_conditioned=not args.disable_frequency_conditioning,
    )
    history_enabled = not bool(args.disable_history_adapter)
    history_adapter = None
    history_history = []
    selected_history_epoch = 0
    history_best_mape = None
    history_baseline_val = None
    history_blend_alpha = 0.0
    history_blend_val = None
    history_context_median = None
    history_context_scale = None
    history_feature_median = None
    history_feature_scale = None
    history_point_cutoff = None
    if history_enabled:
        train_context, train_history_features, train_time = collect_causal_features(
            loaders[0], capacity, adjacency, device, model.daily_len, model.pred_len
        )
        val_context, val_history_features, val_time = collect_causal_features(
            loaders[1], capacity, adjacency, device, model.daily_len, model.pred_len
        )
        history_context_median, history_context_scale = robust_statistics(train_context)
        history_feature_median, history_feature_scale = robust_statistics(
            train_history_features
        )
        # The support boundary is derived from train-only corrected point
        # forecasts.  History corrections are disabled beyond it so an unseen
        # high-flow regime cannot amplify RMSE through extrapolation.
        history_point_cutoff = float(adapter_corrected_train.max().item())
        train_context = robust_normalize(
            train_context, history_context_median, history_context_scale
        )
        val_context = robust_normalize(
            val_context, history_context_median, history_context_scale
        )
        train_history_features = robust_normalize(
            train_history_features, history_feature_median, history_feature_scale
        )
        val_history_features = robust_normalize(
            val_history_features, history_feature_median, history_feature_scale
        )
        (
        history_adapter,
            history_history,
            selected_history_epoch,
            history_best_mape,
            history_baseline_val,
            history_blend_alpha,
            history_blend_val,
        ) = train_history_adapter(
            adapter_corrected_train,
            tuned_train[1],
            point_train_features,
            train_context,
            train_history_features,
            train_time,
            adapter_corrected_val,
            tuned_val[1],
            point_val_features,
            val_context,
            val_history_features,
            val_time,
            args.dataset,
            device,
            args,
            point_cutoff=history_point_cutoff,
        )
        corrected_train = apply_history_adapter(
            history_adapter,
            adapter_corrected_train,
            point_train_features,
            train_context,
            train_history_features,
            train_time,
            device,
            args.batch_size,
            history_blend_alpha,
            point_cutoff=history_point_cutoff,
        )
        corrected_val = apply_history_adapter(
            history_adapter,
            adapter_corrected_val,
            point_val_features,
            val_context,
            val_history_features,
            val_time,
            device,
            args.batch_size,
            history_blend_alpha,
            point_cutoff=history_point_cutoff,
        )
    else:
        corrected_train = adapter_corrected_train
        corrected_val = adapter_corrected_val
    ratio_calibration, ratio_baseline_val, ratio_rows = select_ratio_calibration(
        corrected_train,
        tuned_train[1],
        corrected_val,
        tuned_val[1],
        args.dataset,
        args,
    )
    corrected_train = apply_ratio_calibration(corrected_train, ratio_calibration)
    corrected_val = apply_ratio_calibration(corrected_val, ratio_calibration)
    head_train_features = normalize_features(
        replace_point_features(tuned_train[2], corrected_train),
        feature_median,
        feature_scale,
    )
    head_val_features = normalize_features(
        replace_point_features(tuned_val[2], corrected_val),
        feature_median,
        feature_scale,
    )
    _, basis, pc_std, local_std, explained = fit_basis(
        tuned_train[1] - corrected_train,
        args.pc_rank,
    )
    (
        head,
        head_history,
        best_head_score,
        selected_head_epoch,
        head_baseline,
        head_selection_mode,
        head_eligible_epoch_count,
    ) = train_uncertainty_head(
        corrected_train,
        tuned_train[1],
        head_train_features,
        corrected_val,
        tuned_val[1],
        head_val_features,
        basis,
        pc_std,
        local_std,
        device,
        args,
        orthogonal_two_source=not args.disable_orthogonal_two_source,
        analytic_probabilistic_eval=args.analytic_probabilistic_eval,
    )
    assert_point_path_matches_forward(model, loaders[2], scaler, device)
    evaluation = evaluate(
        model,
        adapter,
        head,
        loaders[2],
        scaler,
        adjacency,
        capacity,
        feature_median,
        feature_scale,
        mean_directions,
        basis,
        pc_std,
        local_std,
        args.dataset,
        device,
        adapter_blend_alpha=adapter_blend_alpha,
        history_adapter=history_adapter,
        history_context_median=history_context_median,
        history_context_scale=history_context_scale,
        history_feature_median=history_feature_median,
        history_feature_scale=history_feature_scale,
        history_blend_alpha=history_blend_alpha,
        history_point_cutoff=history_point_cutoff,
        ratio_calibration=ratio_calibration,
        orthogonal_two_source=not args.disable_orthogonal_two_source,
        mean_consistency=not args.disable_mean_consistency,
        frequency_conditioned=not args.disable_frequency_conditioning,
        analytic_probabilistic_eval=args.analytic_probabilistic_eval,
    )
    if args.analytic_probabilistic_eval:
        totals, means, stds, lowers, uppers, targets, max_dev = evaluation
    else:
        totals, means, lowers, uppers, targets, max_dev = evaluation
        stds = []
    all_targets = np.concatenate(targets, axis=0)[..., None]
    canonical = np.rint(all_targets).astype(np.float32, copy=False)
    canonical[canonical == 0] = 0.0
    metrics = {
        "MAE": totals["MAE"] / totals["COUNT"],
        "RMSE": (totals["SQ"] / totals["COUNT"]) ** 0.5,
        "MAPE": 100.0 * totals["MAPE"] / max(totals["MAPE_COUNT"], 1.0),
        "CRPS": totals["CRPS_NUM"] / totals["CRPS_DEN"],
        "MIS": totals["MIS"] / totals["COUNT"],
        "model": "TOCU",
        "model_alias": "TOCU",
        "dataset": args.dataset,
        "seed": args.seed,
        "status": "completed",
        "training_mode": f"guarded_{args.adapter_mode}_crps",
        "backbone": "balanced_point_fine_tuning_with_decayed_anchor",
        "risk_feature_scale": "original_input_consistent",
        "epochs_requested": args.epochs,
        "epochs_completed": len(backbone_history),
        "selected_backbone_epoch": selected_backbone_epoch,
        "best_point_val_score": best_point_score,
        "backbone_selection_metric": args.backbone_selection_metric,
        "validation_tail_fraction": args.validation_tail_fraction,
        "validation_tail_selection_weight": args.validation_tail_selection_weight,
        "training_sample_profile": getattr(
            train_backbone, "last_sample_profile", {"enabled": False}
        ),
        "baseline_val": baseline_val,
        "adapter_epochs_completed": len(adapter_history),
        "selected_adapter_epoch": selected_adapter_epoch,
        "best_adapter_val_score": best_adapter_score,
        "adapter_baseline_val": adapter_baseline,
        "adapter_blend_alpha": adapter_blend_alpha,
        "adapter_blend_val": adapter_blend_val,
        "adapter_blend_val_score": adapter_blend_score,
        "adapter_blend_selection_metric": args.adapter_blend_selection_metric,
        "adapter_blend_grid": adapter_blend_rows,
        "history_adapter_enabled": history_enabled,
        "history_adapter_epochs_completed": len(history_history),
        "selected_history_epoch": selected_history_epoch,
        "history_best_val_mape": history_best_mape,
        "history_baseline_val": history_baseline_val,
        "history_blend_alpha": history_blend_alpha,
        "history_blend_val": history_blend_val,
        "history_point_cutoff": history_point_cutoff,
        "ratio_calibration_enabled": ratio_calibration is not None,
        "ratio_calibration_mode": getattr(args, "ratio_calibration_mode", "off"),
        "ratio_calibration_baseline_val": ratio_baseline_val,
        "ratio_calibration_selected": (
            None
            if ratio_calibration is None
            else {
                key: value
                for key, value in ratio_calibration.items()
                if key != "ratio"
            }
        ),
        "ratio_calibration_grid": [
            {key: value for key, value in row.items() if key != "ratio"}
            for row in ratio_rows
        ],
        "history_max_delta": args.history_max_delta,
        "history_hidden": args.history_hidden,
        "history_node_dim": args.history_node_dim,
        "history_horizon_dim": args.history_horizon_dim,
        "history_epochs_requested": args.history_epochs,
        "history_patience": args.history_patience,
        "history_lr": args.history_lr,
        "history_weight_decay": args.history_weight_decay,
        "history_log_weight": args.history_log_weight,
        "history_relative_weight": args.history_relative_weight,
        "history_relative_target_power": args.history_relative_target_power,
        "history_mae_weight": args.history_mae_weight,
        "history_rmse_weight": args.history_rmse_weight,
        "history_delta_penalty": args.history_delta_penalty,
        "history_horizon_weight": args.history_horizon_weight,
        "history_guard_tolerance": args.history_guard_tolerance,
        "history_blends": args.history_blends,
        "ratio_guard_tolerance": args.ratio_guard_tolerance,
        "head_epochs_completed": len(head_history),
        "selected_head_epoch": selected_head_epoch,
        "best_head_val_score": best_head_score,
        "head_baseline_val": head_baseline,
        "head_selection_policy": "guarded_crps_with_coverage_fallback",
        "head_selection_mode": head_selection_mode,
        "head_eligible_epoch_count": head_eligible_epoch_count,
        "optimizer": "AdamW",
        "backbone_lr": args.backbone_lr,
        "adapter_lr": args.adapter_lr,
        "head_lr": args.head_lr,
        "rmse_weight": args.rmse_weight,
        "relative_weight": args.relative_weight,
        "tail_weight": args.tail_weight,
        "low_flow_weight": args.low_flow_weight,
        "horizon_weight": args.horizon_weight,
        "low_flow_power": args.low_flow_power,
        "log_relative_weight": args.log_relative_weight,
        "sample_low_flow_weight": args.sample_low_flow_weight,
        "sample_low_flow_power": args.sample_low_flow_power,
        "sample_input_low_flow_weight": args.sample_input_low_flow_weight,
        "sample_input_low_flow_power": args.sample_input_low_flow_power,
        "sample_hard_weight": args.sample_hard_weight,
        "sample_tail_weight": args.sample_tail_weight,
        "sample_tail_power": args.sample_tail_power,
        "sample_weight_clip": args.sample_weight_clip,
        "sample_weight_ramp_power": args.sample_weight_ramp_power,
        "adapter_low_flow_weight": effective_adapter_low_flow_weight,
        "adapter_horizon_weight": effective_adapter_horizon_weight,
        "adapter_low_flow_power": effective_adapter_low_flow_power,
        "adapter_log_relative_weight": effective_adapter_log_relative_weight,
        "adapter_mode": args.adapter_mode,
        "adapter_selection_metric": args.adapter_selection_metric,
        "anchor_start": args.anchor_start,
        "anchor_end": args.anchor_end,
        "guard_tolerance": args.guard_tolerance,
        "adapter_guard_tolerance": args.adapter_guard_tolerance,
        "adapter_blend_tolerance": (
            args.adapter_blend_tolerance
            if args.adapter_blend_tolerance is not None
            else args.adapter_guard_tolerance
        ),
        "head_nll_weight": args.head_nll_weight,
        "head_crps_weight": args.head_crps_weight,
        "coverage_weight": args.coverage_weight,
        "head_nll_tolerance": args.head_nll_tolerance,
        "head_coverage_tolerance": args.head_coverage_tolerance,
        "num_samples": 0 if args.analytic_probabilistic_eval else NUM_SAMPLES,
        "pc_rank": basis.shape[0],
        "pc_explained_variance": explained,
        "mean_direction_count": mean_directions.shape[0],
        "conditioning_features": list(FEATURE_NAMES),
        "max_sample_mean_deviation": max_dev,
        "split": [0.6, 0.2, 0.2],
        "input_len": 12,
        "output_len": 12,
        "innovation_flags": [
            "frequency_conditioned",
            "orthogonal_two_source",
            "frequency_impedance_variance_consistency",
        ],
        "optimization_flags": [
            "mape_and_tail_balanced_point_loss",
            "ripcn_style_directional_mean_adapter",
            "validation_guarded_adapter_blend",
            "crps_aware_uncertainty_selection",
            "low_flow_horizon_weighted_point_loss",
            "optional_log_relative_point_loss",
            "horizon_conditioned_residual_adapter",
            "single_pass_sample_mean_constraint",
            "post_adapter_uncertainty_conditioning",
            "causal_history_temporal_mape_adapter",
            "train_only_low_flow_horizon_ratio_calibration",
        ],
        "metric_scales": {
            "MAE": "original",
            "RMSE": "original",
            "MAPE": "percent",
            "MIS": "original",
            "CRPS": "raw_target_normalized",
        },
        "mape_protocol": "target_gt_10" if args.dataset == "Seattle" else "target_gt_1",
        "crps_protocol": (
            "analytic_gaussian_crps_raw_target_div_sum_abs_target"
            if args.analytic_probabilistic_eval
            else "quantile_pinball_q05_to_q95_raw_target_div_sum_abs_target"
        ),
        "mis_interval": [0.025, 0.975],
        "target_sha256": hashlib.sha256(
            np.ascontiguousarray(canonical).tobytes()
        ).hexdigest(),
        "target_shape": list(canonical.shape),
        "experiment_family": args.experiment_family,
        "experiment_config": args.experiment_config,
        "sampling_protocol": (
            "analytic_marginal_gaussian"
            if args.analytic_probabilistic_eval
            else "independent_structural_local_samples"
        ),
        "analytic_probabilistic_eval": bool(args.analytic_probabilistic_eval),
        "interval_protocol": (
            "analytic_gaussian_central_95"
            if args.analytic_probabilistic_eval
            else "empirical_sample_quantile_0.025_0.975"
        ),
        "sample_mean_correction_applied": bool(
            (not args.disable_mean_consistency) and not args.analytic_probabilistic_eval
        ),
        "frequency_gate_signal": "normalized_frequency_energy_plus_impedance_risk",
        "frequency_conditioning_enabled": not args.disable_frequency_conditioning,
        "orthogonal_two_source_enabled": not args.disable_orthogonal_two_source,
        "mean_consistency_enabled": not args.disable_mean_consistency,
        "point_path_ablation": {
            "frequency_conditioned": not args.disable_frequency_conditioning,
            "orthogonal_two_source": not args.disable_orthogonal_two_source,
            "mean_consistency": not args.disable_mean_consistency,
        },
        "ablation_variant": args.ablation_variant,
        "innovation_module_status": {
            "frequency_conditioned": not args.disable_frequency_conditioning,
            "orthogonal_two_source": not args.disable_orthogonal_two_source,
            "frequency_impedance_variance_consistency": not args.disable_mean_consistency,
        },
        "wall_time_seconds": time.perf_counter() - started_at,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if torch.cuda.is_available() else 0.0
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (output / "history.json").write_text(
        json.dumps(
        {
            "backbone": backbone_history,
            "adapter": adapter_history,
            "history": history_history,
            "head": head_history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "adapter_state_dict": adapter.state_dict(),
            "head_state_dict": head.state_dict(),
            "feature_median": feature_median,
            "feature_scale": feature_scale,
            "mean_directions": mean_directions,
            "basis": basis,
            "pc_std": pc_std,
            "local_std": local_std,
            "adapter_blend_alpha": adapter_blend_alpha,
            "history_adapter_state_dict": (
                None
                if history_adapter is None
                else history_adapter.state_dict()
            ),
            "history_context_median": history_context_median,
            "history_context_scale": history_context_scale,
            "history_feature_median": history_feature_median,
            "history_feature_scale": history_feature_scale,
            "history_blend_alpha": history_blend_alpha,
            "history_point_cutoff": history_point_cutoff,
            "ratio_calibration": ratio_calibration,
        },
        output / "checkpoint.pt",
    )
    summary_arrays = {
        "mean": np.concatenate(means)[..., None],
        "lower": np.concatenate(lowers)[..., None],
        "upper": np.concatenate(uppers)[..., None],
        "target": all_targets,
    }
    if args.analytic_probabilistic_eval:
        if not stds:
            raise RuntimeError("analytic evaluation did not produce standard deviations")
        summary_arrays["std"] = np.concatenate(stds)[..., None]
    np.savez_compressed(output / "summary.npz", **summary_arrays)
    (output / "DONE").write_text("completed\n", encoding="utf-8")
    print(json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--adapter-epochs", type=int, default=80)
    parser.add_argument("--adapter-patience", type=int, default=12)
    parser.add_argument("--head-epochs", type=int, default=100)
    parser.add_argument("--head-patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--pc-rank", type=int, default=3)
    parser.add_argument("--backbone-lr", type=float, default=2e-6)
    parser.add_argument("--adapter-lr", type=float, default=5e-3)
    parser.add_argument("--head-lr", type=float, default=1e-2)
    parser.add_argument("--rmse-weight", type=float, default=0.5)
    parser.add_argument("--relative-weight", type=float, default=0.35)
    parser.add_argument("--tail-weight", type=float, default=0.35)
    parser.add_argument("--low-flow-weight", type=float, default=0.0)
    parser.add_argument("--horizon-weight", type=float, default=0.0)
    parser.add_argument("--low-flow-power", type=float, default=0.5)
    parser.add_argument(
        "--sample-low-flow-weight",
        type=float,
        default=0.0,
        help="Train-only window-level low-target replay weight; zero preserves the base objective.",
    )
    parser.add_argument("--sample-low-flow-power", type=float, default=0.5)
    parser.add_argument("--sample-input-low-flow-weight", type=float, default=0.0)
    parser.add_argument("--sample-input-low-flow-power", type=float, default=0.5)
    parser.add_argument("--sample-hard-weight", type=float, default=0.0)
    parser.add_argument("--sample-tail-weight", type=float, default=0.0)
    parser.add_argument("--sample-tail-power", type=float, default=0.5)
    parser.add_argument("--sample-weight-clip", type=float, default=3.0)
    parser.add_argument("--sample-weight-ramp-power", type=float, default=1.0)
    parser.add_argument("--validation-tail-fraction", type=float, default=0.0)
    parser.add_argument(
        "--validation-tail-selection-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--log-relative-weight",
        type=float,
        default=0.0,
        help="Optional log1p-relative point-loss weight; zero preserves the base behavior.",
    )
    parser.add_argument("--adapter-log-relative-weight", type=float, default=None)
    parser.add_argument("--adapter-low-flow-weight", type=float, default=None)
    parser.add_argument("--adapter-horizon-weight", type=float, default=None)
    parser.add_argument("--adapter-low-flow-power", type=float, default=None)
    parser.add_argument(
        "--adapter-mode", choices=("directional", "horizon"), default="directional"
    )
    parser.add_argument("--anchor-start", type=float, default=0.5)
    parser.add_argument("--anchor-end", type=float, default=0.1)
    parser.add_argument("--adapter-penalty", type=float, default=0.02)
    parser.add_argument("--guard-tolerance", type=float, default=0.001)
    parser.add_argument("--adapter-guard-tolerance", type=float, default=0.0)
    parser.add_argument("--adapter-blend-tolerance", type=float, default=None)
    parser.add_argument(
        "--adapter-selection-metric",
        choices=("score", "mape", "mae", "rmse"),
        default="score",
    )
    parser.add_argument(
        "--backbone-selection-metric",
        choices=("score", "mape", "mae", "rmse"),
        default="score",
    )
    parser.add_argument(
        "--adapter-blend-selection-metric",
        choices=("score", "mape", "mae", "rmse"),
        default="score",
    )
    parser.add_argument("--rmse-score-weight", type=float, default=1.25)
    parser.add_argument("--mape-score-weight", type=float, default=1.25)
    parser.add_argument("--head-nll-weight", type=float, default=0.02)
    parser.add_argument("--head-crps-weight", type=float, default=1.0)
    parser.add_argument("--coverage-weight", type=float, default=0.2)
    parser.add_argument("--head-nll-tolerance", type=float, default=0.02)
    parser.add_argument("--head-coverage-tolerance", type=float, default=0.03)
    parser.add_argument(
        "--disable-history-adapter",
        action="store_true",
        help="Disable the causal history-temporal point adapter for compatibility probes.",
    )
    parser.add_argument("--ablation-variant", default="full")
    parser.add_argument(
        "--experiment-family",
        choices=("ablation", "hyperparam", "efficiency"),
        default="ablation",
    )
    parser.add_argument("--experiment-config", default=None)
    parser.add_argument("--disable-frequency-conditioning", action="store_true")
    parser.add_argument("--disable-orthogonal-two-source", action="store_true")
    parser.add_argument("--disable-mean-consistency", action="store_true")
    parser.add_argument(
        "--analytic-probabilistic-eval",
        action="store_true",
        help="Evaluate the fitted two-source Gaussian marginal analytically and save std.",
    )
    parser.add_argument("--history-epochs", type=int, default=80)
    parser.add_argument("--history-patience", type=int, default=16)
    parser.add_argument("--history-lr", type=float, default=1.5e-3)
    parser.add_argument("--history-weight-decay", type=float, default=1e-4)
    parser.add_argument("--history-hidden", type=int, default=128)
    parser.add_argument("--history-node-dim", type=int, default=24)
    parser.add_argument("--history-horizon-dim", type=int, default=16)
    parser.add_argument("--history-max-delta", type=float, default=0.60)
    parser.add_argument("--history-log-weight", type=float, default=0.10)
    parser.add_argument("--history-relative-weight", type=float, default=4.0)
    parser.add_argument("--history-relative-target-power", type=float, default=1.0)
    parser.add_argument("--history-mae-weight", type=float, default=0.10)
    parser.add_argument("--history-rmse-weight", type=float, default=0.10)
    parser.add_argument("--history-delta-penalty", type=float, default=0.005)
    parser.add_argument("--history-horizon-weight", type=float, default=0.25)
    parser.add_argument("--history-guard-tolerance", type=float, default=0.001)
    parser.add_argument(
        "--ratio-calibration-mode",
        choices=("off", "horizon", "global_horizon"),
        default="off",
        help="Enable train-only low-flow multiplicative calibration after the history adapter.",
    )
    parser.add_argument("--ratio-guard-tolerance", type=float, default=0.001)
    parser.add_argument(
        "--history-blends",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 1.0],
    )
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
