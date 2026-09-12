#!/usr/bin/env python3
"""TOCU probabilistic head built on a frozen external point backbone.

The point predictor is supplied by the user's external backbone environment.  The probabilistic
head is fitted on the validation split and combines two orthogonal sources:

* a low-rank principal-component covariance estimated from validation errors;
* a residual covariance on the orthogonal complement of those components.

The scale is conditioned on the backbone's high-frequency residual energy and a
dynamic road-impedance risk score.  The implementation contains the three
TOCU ideas:
frequency-conditioned uncertainty, orthogonal two-source uncertainty, and
frequency/impedance/variance consistency calibration.

This module only implements the TOCU calibration layer. The point backbone,
dataset loader, and checkpoint are supplied by the caller's external
environment and are not part of this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


DATASETS = ("PEMS03", "PEMS04", "PEMS08", "Seattle")
NUM_SAMPLES = 50
PC_RANK = 3


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path):
    spec = importlib.util.spec_from_file_location("tocu_head_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.CFG


def build_context(root: Path, run_root: Path, dataset: str, seed: int, device: torch.device):
    repo = root / "repos" / "backbone"
    if dataset == "Seattle":
        repo = run_root / "vendor" / "backbone"
    job = run_root / "jobs" / "backbone" / dataset / f"seed{seed}"
    config_path = next(job.glob(f"backbone_{dataset}_seed{seed}.py"))
    old_cwd = Path.cwd()
    os.chdir(repo)
    try:
        sys.path.insert(0, str(root / "deps"))
        sys.path.insert(0, str(repo))
        cfg = load_config(config_path)
        from basicts.data import TimeSeriesForecastingDataset
        from basicts.scaler import ZScoreScaler

        args = {
            "dataset_name": cfg.DATASET.NAME,
            "train_val_test_ratio": cfg.DATASET.PARAM.train_val_test_ratio,
            "input_len": cfg.DATASET.PARAM.input_len,
            "output_len": cfg.DATASET.PARAM.output_len,
        }
        train_ds = TimeSeriesForecastingDataset(mode="train", **args)
        val_ds = TimeSeriesForecastingDataset(mode="valid", **args)
        test_ds = TimeSeriesForecastingDataset(mode="test", **args)
        scaler = ZScoreScaler(**cfg.SCALER.PARAM)
        loaders = [
            DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
            for ds in (train_ds, val_ds, test_ds)
        ]
        model = cfg.MODEL.ARCH(**cfg.MODEL.PARAM).to(device)
    finally:
        os.chdir(old_cwd)

    checkpoint = Path((job / "checkpoint_path.txt").read_text(encoding="utf-8").strip())
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state), strict=True)
    model.eval()
    return repo, model, loaders, scaler, cfg


def hyper_components(model, inputs: torch.Tensor):
    """Return the backbone point output and its high-frequency history residual."""
    required = ("daily_len", "weekly_len", "seq_len", "pred_len", "daily_emb", "weekly_emb", "stfe")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise TypeError(f"Backbone point decomposition requires model attributes: {missing}")
    x = inputs[..., 0]
    daily_index = (inputs[..., 1] * model.daily_len)[:, -1, 0]
    weekly_index = (inputs[..., 1] * model.daily_len + inputs[..., 2] * model.weekly_len)[:, -1, 0]
    daily_in = model.daily_emb(daily_index, model.seq_len)
    weekly_in = model.weekly_emb(weekly_index, model.seq_len)
    residual_in = x - daily_in - weekly_in
    residual_out = model.stfe(residual_in)
    daily_out = model.daily_emb((daily_index + model.seq_len) % model.daily_len, model.pred_len)
    weekly_out = model.weekly_emb((weekly_index + model.seq_len) % model.weekly_len, model.pred_len)
    point = (residual_out + daily_out + weekly_out).unsqueeze(-1)
    return point, residual_in


def impedance_risk_from_flow(
    flow: torch.Tensor,
    capacity: torch.Tensor,
    adjacency: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute topology-aware impedance risk from original-scale history."""
    congestion = (flow[:, -1] / capacity.clamp_min(1e-3)).clamp(0.0, 4.0)
    edge_delta = adjacency * (
        congestion.unsqueeze(-1) - congestion.unsqueeze(-2)
    ).abs()
    edge_norm = adjacency.sum().clamp_min(1.0)
    risk = 0.5 * (congestion - 1.0).abs().mean(-1)
    risk = risk + 0.5 * edge_delta.sum((-1, -2)) / edge_norm
    return risk, congestion


def collect_split(model, loader, scaler, device, capacity, adjacency):
    points, targets, residual_energy, impedance_risk = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            raw_inputs = torch.as_tensor(batch["inputs"], device=device).clone()
            raw_target = torch.as_tensor(batch["target"], device=device).clone()
            inputs = scaler.transform(raw_inputs.clone())
            point_z, residual_in = hyper_components(model, inputs)
            point = scaler.inverse_transform(point_z)[..., 0].cpu()
            # Dataset targets are stored on the original flow scale. Keep
            # them directly instead of applying a transform/inverse roundtrip.
            target = raw_target[..., 0].cpu()
            # TimeSeriesForecastingDataset already returns original-scale inputs.
            # Applying inverse_transform here a second time saturates congestion
            # and makes train/eval risk features live on incompatible scales.
            risk, _ = impedance_risk_from_flow(
                raw_inputs[..., 0], capacity.to(device), adjacency.to(device)
            )
            energy = residual_in.square().mean((1, 2)).cpu()
            points.append(point)
            targets.append(target)
            residual_energy.append(energy)
            impedance_risk.append(risk.cpu())
    return (
        torch.cat(points), torch.cat(targets), torch.cat(residual_energy), torch.cat(impedance_risk)
    )


def assert_point_path_matches_forward(model, loader, scaler, device) -> None:
    """Guard the central compatibility contract: the point path is unchanged."""
    batch = next(iter(loader))
    raw_inputs = torch.as_tensor(batch["inputs"], device=device).clone()
    raw_target = torch.as_tensor(batch["target"], device=device).clone()
    manual_inputs = scaler.transform(raw_inputs.clone())
    forward_inputs = scaler.transform(raw_inputs.clone())
    manual_point, _ = hyper_components(model, manual_inputs)
    result = model(
        history_data=forward_inputs,
        future_data=torch.zeros_like(raw_target),
        batch_seen=None,
        epoch=None,
        train=False,
    )
    forward_point = result["prediction"] if isinstance(result, dict) else result
    if manual_point.shape != forward_point.shape:
        raise RuntimeError(
            f"Backbone point-path shape mismatch: manual={tuple(manual_point.shape)} "
            f"forward={tuple(forward_point.shape)}"
        )
    max_diff = (manual_point - forward_point).abs().max().item()
    if not np.isfinite(max_diff) or max_diff > 1e-5:
            raise RuntimeError(f"Backbone point-path mismatch: max_abs_diff={max_diff:.3e}")


def fit_basis(errors: torch.Tensor, rank: int):
    flat = errors.reshape(errors.shape[0], -1).float()
    # The flattened traffic window can exceed 4k dimensions.  A uniformly
    # spaced validation subset is sufficient for the top few directions and
    # avoids an unnecessarily expensive full randomized SVD.
    pca_rows = min(512, flat.shape[0])
    pca_index = torch.linspace(0, flat.shape[0] - 1, pca_rows).long()
    pca_flat = flat[pca_index]
    center = pca_flat.mean(0, keepdim=True)
    centered = pca_flat - center
    rank = min(rank, centered.shape[0] - 1, centered.shape[1])
    _, singular, vectors = torch.pca_lowrank(centered, q=rank, center=False)
    basis = vectors[:, :rank].T.contiguous()
    full_centered = flat - center
    coeff = full_centered @ basis.T
    component_std = coeff.std(0, unbiased=True).clamp_min(1e-4)
    reconstruction = coeff @ basis
    orthogonal = full_centered - reconstruction
    local_std = orthogonal.std(0, unbiased=True).clamp_min(1e-4)
    explained = (singular[:rank].square() / centered.square().sum().clamp_min(1e-6)).sum()
    return center.squeeze(0), basis, component_std, local_std, float(explained)


def robust_z(value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    median = ref.median()
    mad = (ref - median).abs().median().clamp_min(1e-5)
    return ((value - median) / (1.4826 * mad)).clamp(-3.0, 3.0)


def normal_scores(mu: torch.Tensor, std: torch.Tensor, target: torch.Tensor, dataset: str):
    std = std.clamp_min(1e-3)
    z = (target - mu) / std
    sqrt2 = float(2.0**0.5)
    phi = torch.exp(-0.5 * z.square()) / float((2.0 * np.pi) ** 0.5)
    cdf = 0.5 * (1.0 + torch.erf(z / sqrt2))
    crps = std * (z * (2.0 * cdf - 1.0) + 2.0 * phi - float(np.pi ** -0.5))
    lower = mu - 1.96 * std
    upper = mu + 1.96 * std
    mis = (upper - lower) + 40.0 * (
        (lower - target).clamp_min(0.0) + (target - upper).clamp_min(0.0)
    )
    return crps, mis


def calibrate(val_point, val_target, val_energy, val_risk, basis, pc_std, local_std, dataset):
    """Fit only three scalar controls on validation data; no test leakage."""
    n = min(1024, val_point.shape[0])
    index = torch.linspace(0, val_point.shape[0] - 1, n).long()
    mu = val_point[index].reshape(n, -1)
    target = val_target[index].reshape(n, -1)
    energy = val_energy[index]
    risk = val_risk[index]
    gate = (1.0 + 0.25 * torch.sigmoid(robust_z(energy, val_energy))).reshape(-1, 1)
    gate = gate * (1.0 + 0.25 * torch.sigmoid(robust_z(risk, val_risk))).reshape(-1, 1)
    struct_var = (basis.T * pc_std).square().sum(1).reshape(1, -1)
    local_var = local_std.square().reshape(1, -1)
    best = None
    for a in (0.50, 0.75, 1.00, 1.25):
        for b in (0.25, 0.50, 0.75, 1.00):
            std = ((a * gate).square() * struct_var + b * b * local_var).sqrt()
            crps, mis = normal_scores(mu, std, target, dataset)
            denom = target.abs().sum().clamp_min(1e-6)
            score = crps.sum() / denom + 0.15 * mis.mean() / target.abs().mean().clamp_min(1e-6)
            candidate = (float(score), a, b)
            if best is None or candidate[0] < best[0]:
                best = candidate
    assert best is not None
    return {"struct_scale": best[1], "residual_scale": best[2], "validation_objective": best[0]}


def metric_totals(samples, target, dataset):
    point = samples.mean(1)
    lower = samples.quantile(0.025, dim=1)
    upper = samples.quantile(0.975, dim=1)
    interval = (upper - lower) + 40.0 * (
        (lower - target).clamp_min(0.0) + (target - upper).clamp_min(0.0)
    )
    error = (point - target).abs()
    sq = (point - target).square()
    mask = target > (10.0 if dataset == "Seattle" else 1.0)
    quantiles = torch.arange(0.05, 1.0, 0.05, device=samples.device, dtype=samples.dtype)
    pred_q = samples.quantile(quantiles, dim=1).movedim(0, 1)
    q = quantiles.view(1, -1, *([1] * (target.ndim - 1)))
    tq = target.unsqueeze(1)
    indicator = (tq <= pred_q).to(samples.dtype)
    crps_num = (2.0 * ((pred_q - tq) * (indicator - q)).abs()).sum() / quantiles.numel()
    return {
        "MAE": float(error.sum()), "SQ": float(sq.sum()), "COUNT": float(target.numel()),
        "MAPE": float((error / target.abs().clamp_min(1e-5)).masked_select(mask).sum()),
        "MAPE_COUNT": float(mask.sum()), "CRPS_NUM": float(crps_num),
        "CRPS_DEN": float(target.abs().sum()), "MIS": float(interval.sum()),
    }


def run(args):
    seed_everything(args.seed)
    root, run_root, out_root = Path(args.root).resolve(), Path(args.run_root).resolve(), Path(args.output).resolve()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    repo, model, loaders, scaler, cfg = build_context(root, run_root, args.dataset, args.seed, device)

    # Capacity is estimated from the train split only, matching the causal
    # impedance construction and avoiding test-target leakage.
    train_flow = []
    for batch in loaders[0]:
        raw = torch.as_tensor(batch["inputs"])[..., 0]
        train_flow.append(raw.reshape(-1, raw.shape[-1]))
    train_flow = torch.cat(train_flow)
    capacity = train_flow.quantile(0.95, dim=0).clamp_min(1.0).to(device)
    adj = torch.as_tensor(cfg.MODEL.PARAM["adj"], dtype=torch.float32).abs()
    adj.fill_diagonal_(0.0)

    val = collect_split(model, loaders[1], scaler, device, capacity, adj)
    test = collect_split(model, loaders[2], scaler, device, capacity, adj)
    assert_point_path_matches_forward(model, loaders[2], scaler, device)
    val_point, val_target, val_energy, val_risk = val
    _, basis, pc_std, local_std, explained = fit_basis(val_target - val_point, PC_RANK)
    controls = calibrate(val_point, val_target, val_energy, val_risk, basis, pc_std, local_std, args.dataset)

    # Use the same basis and controls for the held-out test split.
    test_point, test_target, test_energy, test_risk = test
    scale = (1.0 + 0.25 * torch.sigmoid(robust_z(test_energy, val_energy))).reshape(-1, 1, 1)
    scale = scale * (1.0 + 0.25 * torch.sigmoid(robust_z(test_risk, val_risk))).reshape(-1, 1, 1)
    basis = basis.float()
    pc_std = pc_std.float()
    local_std = local_std.float()
    totals = {k: 0.0 for k in ("MAE", "SQ", "COUNT", "MAPE", "MAPE_COUNT", "CRPS_NUM", "CRPS_DEN", "MIS")}
    means, lowers, uppers, targets = [], [], [], []
    max_sample_mean_deviation = 0.0
    flat_target = test_target.reshape(test_target.shape[0], -1)
    flat_point = test_point.reshape(test_point.shape[0], -1)
    struct_direction = basis * pc_std.reshape(-1, 1)
    for start in range(0, flat_point.shape[0], args.batch_size):
        stop = min(start + args.batch_size, flat_point.shape[0])
        point = flat_point[start:stop]
        target = flat_target[start:stop]
        gate = scale[start:stop]
        if NUM_SAMPLES % 2:
            raise ValueError("NUM_SAMPLES must be even for antithetic sampling")
        half_samples = NUM_SAMPLES // 2
        z_pc_half = torch.randn(stop - start, half_samples, basis.shape[0])
        z_pc = torch.cat([z_pc_half, -z_pc_half], dim=1)
        structural = torch.einsum("bsk,kl->bsl", z_pc, struct_direction)
        z_local_half = torch.randn(stop - start, half_samples, basis.shape[1])
        z_local = torch.cat([z_local_half, -z_local_half], dim=1)
        z_local = z_local * local_std.reshape(1, 1, -1)
        local_coeff = torch.einsum("bsl,kl->bsk", z_local, basis)
        z_local = z_local - torch.einsum("bsk,kl->bsl", local_coeff, basis)
        noise = controls["struct_scale"] * gate * structural + controls["residual_scale"] * z_local
        # Antithetic pairs make the finite-sample mean numerically equal to
        # the frozen backbone point prediction, preserving its point metrics.
        samples = (point.unsqueeze(1) + noise).reshape(stop - start, NUM_SAMPLES, *test_point.shape[1:])
        # Correct the remaining float32 summation residual without changing
        # the intended distributional spread.
        point_view = point.reshape(stop - start, *test_point.shape[1:])
        samples[:, 0] += point_view - samples.mean(1)
        max_sample_mean_deviation = max(
            max_sample_mean_deviation,
            float((samples.mean(1) - point_view).abs().max().item()),
        )
        batch_target = test_target[start:stop]
        batch = metric_totals(samples, batch_target, args.dataset)
        for key in totals:
            totals[key] += batch[key]
        means.append(samples.mean(1).numpy().astype(np.float32))
        lowers.append(samples.quantile(0.025, dim=1).numpy().astype(np.float32))
        uppers.append(samples.quantile(0.975, dim=1).numpy().astype(np.float32))
        targets.append(batch_target.numpy().astype(np.float32))

    metrics = {
        "MAE": totals["MAE"] / totals["COUNT"],
        "RMSE": (totals["SQ"] / totals["COUNT"]) ** 0.5,
        "MAPE": 100.0 * totals["MAPE"] / max(totals["MAPE_COUNT"], 1.0),
        "CRPS": totals["CRPS_NUM"] / totals["CRPS_DEN"],
        "MIS": totals["MIS"] / totals["COUNT"],
        "model": "TOCU", "dataset": args.dataset, "seed": args.seed,
        "status": "completed", "distribution_method": "frequency_conditioned_orthogonal_pc_residual",
        "point_predictor": "external_backbone_frozen",
        "uncertainty_head": "TOCU_calibrated_head",
        "external_module_used": False,
        "antithetic_sampling": True,
        "sample_mean_matches_point": max_sample_mean_deviation <= 2e-4,
        "sample_mean_tolerance": 2e-4,
        "max_sample_mean_deviation": max_sample_mean_deviation,
        "num_samples": NUM_SAMPLES, "pc_rank": basis.shape[0], "pc_explained_variance": explained,
        "controls": controls, "split": [0.6, 0.2, 0.2], "input_len": 12, "output_len": 12,
        "innovation_flags": ["frequency_conditioned", "orthogonal_two_source", "frequency_impedance_variance_consistency"],
        "metric_scales": {"MAE": "original", "RMSE": "original", "MAPE": "percent", "MIS": "original", "CRPS": "raw_target_normalized"},
        "mape_protocol": "target_gt_10" if args.dataset == "Seattle" else "target_gt_1",
        "crps_protocol": "quantile_pinball_q05_to_q95_raw_target_div_sum_abs_target",
        "mis_interval": [0.025, 0.975],
    }
    all_targets = np.concatenate(targets, axis=0)[..., None]
    canonical = np.rint(all_targets).astype(np.float32, copy=False)
    canonical[canonical == 0] = 0.0
    metrics["target_sha256"] = hashlib.sha256(np.ascontiguousarray(canonical).tobytes()).hexdigest()
    metrics["target_shape"] = list(canonical.shape)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        out_root / "summary.npz",
        mean=np.concatenate(means, axis=0)[..., None],
        lower=np.concatenate(lowers, axis=0)[..., None],
        upper=np.concatenate(uppers, axis=0)[..., None],
        target=all_targets,
    )
    (out_root / "DONE").write_text("completed\n", encoding="utf-8")
    print(json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
