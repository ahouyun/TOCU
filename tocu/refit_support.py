#!/usr/bin/env python3
"""TOCU refit support for train-plus-validation forecast-bin calibration.

The external point backbone and directional adapter are frozen from
the audited source run.  The causal history adapter is validation-selected on
train/validation, then refit for the selected epoch on train+validation.  A
ratio rule selected by the independent validation-only probe is refit on
the same train+validation point forecasts.  Finally the residual basis and
CRPS head are fit on train+validation for a fixed number of epochs inherited
from the completed source head run.  Test labels are read only for final
descriptive evaluation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import tocu.train as tocu
from tocu.head import (
    NUM_SAMPLES,
    build_context,
    fit_basis,
    seed_everything,
)
from tocu.history import (
    _loss as history_loss,
    CausalHistoryTemporalAdapter,
    causal_context_features,
    collect_causal_features,
    future_time_features,
    history_features,
    robust_normalize,
    robust_statistics,
    train_history_adapter,
)


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def target_hash(target: torch.Tensor) -> dict[str, object]:
    value = target.detach().cpu().numpy().astype(np.float64, copy=False)
    canonical = np.rint(value).astype(np.float32, copy=False)
    return {
        "raw_sha256": sha256_array(value),
        "canonical_sha256": sha256_array(canonical),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def build_frozen_adapter(model, checkpoint, metrics, device):
    feature_count = int(checkpoint["feature_median"].numel())
    direction_scales = checkpoint["adapter_state_dict"]["direction_scales"]
    if metrics.get("adapter_mode") == "horizon":
        adapter = tocu.HorizonDirectionalAdapter(
            feature_count, direction_scales, model.pred_len
        ).to(device)
    else:
        adapter = tocu.DirectionalMeanAdapter(feature_count, direction_scales).to(device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    adapter.eval()
    return adapter


def collect_frozen_split(
    model,
    loader,
    scaler,
    adjacency,
    capacity,
    adapter,
    checkpoint,
    metrics,
    device,
    include_graph_history: bool = False,
    history_graph_mode: str = "outgoing",
):
    """Collect the audited source point path plus causal features in loader order."""
    points, targets, raw_features = tocu.collect_split(
        model, loader, scaler, device, capacity, adjacency
    )
    feature_median = checkpoint["feature_median"]
    feature_scale = checkpoint["feature_scale"]
    features = tocu.normalize_features(raw_features, feature_median, feature_scale)
    corrected = tocu.apply_adapter(
        adapter,
        points,
        features,
        checkpoint["mean_directions"],
        device,
        128,
        blend_alpha=float(metrics.get("adapter_blend_alpha", 1.0)),
    )
    context, history, times = collect_causal_features(
        loader, capacity, adjacency, device, model.daily_len, model.pred_len,
        include_graph_history=include_graph_history,
        graph_mode=history_graph_mode,
    )
    return {
        "point": corrected,
        "target": targets,
        "raw_features": raw_features,
        "context": context,
        "history": history,
        "times": times,
    }


def normalize_history_values(split, context_median, context_scale, history_median, history_scale):
    return {
        **split,
        "global": tocu.normalize_features(
            tocu.replace_point_features(split["raw_features"], split["point"]),
            split["feature_median"],
            split["feature_scale"],
        ),
        "context_norm": robust_normalize(
            split["context"], context_median, context_scale
        ),
        "history_norm": robust_normalize(
            split["history"], history_median, history_scale
        ),
    }


def fit_history_fixed(values, device, args, epochs: int):
    """Refit the selected history architecture for a frozen epoch count."""
    point, target, global_features, context, history, times = values
    tocu.seed_everything(int(args.seed) + 9101)
    adapter = CausalHistoryTemporalAdapter(
        global_features.shape[1],
        context.shape[2],
        history.shape[2],
        times.shape[2],
        point.shape[1],
        point.shape[2],
        hidden=int(args.history_hidden),
        node_dim=int(args.history_node_dim),
        horizon_dim=int(args.history_horizon_dim),
        max_delta=float(args.history_max_delta),
    ).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(args.history_lr),
        weight_decay=float(args.history_weight_decay),
    )
    generator = torch.Generator().manual_seed(int(args.seed) + 9101)
    history_log = []
    for epoch in range(1, int(epochs) + 1):
        adapter.train()
        order = torch.randperm(point.shape[0], generator=generator)
        losses = []
        for start in range(0, order.numel(), int(args.batch_size)):
            index = order[start : start + int(args.batch_size)]
            corrected, delta = adapter(
                point[index].to(device),
                global_features[index].to(device),
                context[index].to(device),
                history[index].to(device),
                times[index].to(device),
            )
            loss = history_loss(
                corrected,
                target[index].to(device),
                delta,
                args.dataset,
                args.history_log_weight,
                args.history_relative_weight,
                args.history_relative_target_power,
                args.history_mae_weight,
                args.history_rmse_weight,
                args.history_delta_penalty,
                args.history_horizon_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history_log.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
    adapter.eval()
    return adapter, history_log


def apply_history(adapter, values, device, batch_size, point_cutoff):
    outputs = []
    adapter.eval()
    point = values[0]
    with torch.no_grad():
        for start in range(0, point.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), point.shape[0])
            corrected, _ = adapter(
                point[start:stop].to(device),
                values[2][start:stop].to(device),
                values[3][start:stop].to(device),
                values[4][start:stop].to(device),
                values[5][start:stop].to(device),
            )
            base = point[start:stop].to(device)
            if point_cutoff is not None:
                gate = (base < float(point_cutoff)).to(base.dtype)
                corrected = base + gate * (corrected - base)
            outputs.append(corrected.cpu())
    return torch.cat(outputs)


def fit_head_fixed(
    train_point,
    train_target,
    train_features,
    basis,
    pc_std,
    local_std,
    device,
    args,
    epochs: int,
):
    struct_var = (basis.T * pc_std.reshape(1, -1)).square().sum(1)
    local_var = local_std.square()
    head = tocu.CRPSUncertaintyHead(
        train_features.shape[1], struct_var.to(device), local_var.to(device)
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=1e-4)
    tocu.seed_everything(int(args.seed) + 16161)
    generator = torch.Generator().manual_seed(int(args.seed) + 16161)
    logs = []
    for epoch in range(1, int(epochs) + 1):
        head.train()
        order = torch.randperm(train_point.shape[0], generator=generator)
        losses = []
        for start in range(0, order.numel(), int(args.batch_size)):
            index = order[start : start + int(args.batch_size)]
            point = train_point[index].reshape(index.numel(), -1).to(device)
            target = train_target[index].reshape(index.numel(), -1).to(device)
            features = train_features[index].to(device)
            std = head.std(features)
            residual = target - point
            nll = 0.5 * ((residual / std).square() + 2.0 * std.log()).mean()
            scale = target.abs().mean().clamp_min(1.0)
            crps = tocu.gaussian_crps(point, std, target).mean() / scale
            loss = args.head_nll_weight * nll + args.head_crps_weight * crps
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        logs.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "eligible": True})
    head.eval()
    return head, logs


def make_args(args, source_metrics):
    values = vars(args).copy()
    values.update(
        {
            "history_hidden": int(source_metrics.get("history_hidden", 128)),
            "history_node_dim": int(source_metrics.get("history_node_dim", 24)),
            "history_horizon_dim": int(source_metrics.get("history_horizon_dim", 16)),
            "history_max_delta": float(source_metrics.get("history_max_delta", 0.60)),
            "history_log_weight": float(source_metrics.get("history_log_weight", 0.10)),
            "history_relative_weight": float(source_metrics.get("history_relative_weight", 4.0)),
            "history_relative_target_power": float(source_metrics.get("history_relative_target_power", 1.0)),
            "history_mae_weight": float(source_metrics.get("history_mae_weight", 0.10)),
            "history_rmse_weight": float(source_metrics.get("history_rmse_weight", 0.10)),
            "history_delta_penalty": float(source_metrics.get("history_delta_penalty", 0.005)),
            "history_horizon_weight": float(source_metrics.get("history_horizon_weight", 0.25)),
            "history_lr": float(source_metrics.get("history_lr", 1.5e-3)),
            "history_weight_decay": float(source_metrics.get("history_weight_decay", 1e-4)),
            "head_lr": float(source_metrics.get("head_lr", 5e-3)),
            "head_nll_weight": float(source_metrics.get("head_nll_weight", 0.02)),
            "head_crps_weight": float(source_metrics.get("head_crps_weight", 1.0)),
        }
    )
    return SimpleNamespace(**values)


def fit_pred_bin_horizon_ratio(
    prediction: torch.Tensor,
    target: torch.Tensor,
    dataset: str,
    bin_count: int,
    relative_power: float,
    cutoff: float,
    ridge: float,
    device: torch.device,
    gate: str = "below",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit train+validation horizon/forecast-bin ratios without test access."""
    if gate not in {"below", "above"}:
        raise ValueError(f"unsupported ratio calibration gate: {gate}")
    threshold = 10.0 if dataset == "Seattle" else 1.0
    work_prediction = prediction.to(device=device, dtype=torch.float32)
    work_target = target.to(device=device, dtype=torch.float32)
    quantiles = torch.linspace(0.0, 1.0, int(bin_count) + 1, device=device)
    ratios = []
    edges = []
    for horizon in range(work_prediction.shape[1]):
        point = work_prediction[:, horizon]
        truth = work_target[:, horizon]
        edge = torch.quantile(point.reshape(-1), quantiles)
        bins = torch.bucketize(point.contiguous(), edge[1:-1], right=True)
        weight = torch.maximum(truth.abs(), torch.as_tensor(threshold, device=device))
        weight = weight.pow(-float(relative_power))
        if gate == "below":
            supported = point < float(cutoff)
        else:
            supported = point >= float(cutoff)
        weight = weight * supported.to(weight.dtype)
        numerator = weight * point * truth
        denominator = weight * point.square()
        numerator_sum = torch.zeros(int(bin_count), device=device, dtype=torch.float32)
        denominator_sum = torch.zeros(int(bin_count), device=device, dtype=torch.float32)
        numerator_sum.scatter_add_(0, bins.reshape(-1), numerator.reshape(-1))
        denominator_sum.scatter_add_(0, bins.reshape(-1), denominator.reshape(-1))
        ratios.append((numerator_sum + float(ridge)) / (denominator_sum + float(ridge)))
        edges.append(edge)
    return torch.stack(ratios).cpu(), torch.stack(edges).cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--ratio-probe", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--history-selection-epochs", type=int, default=80)
    parser.add_argument("--history-selection-patience", type=int, default=16)
    parser.add_argument("--head-refit-epochs", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--protocol-label",
        default="formal",
        help="Label recorded in the output provenance; does not alter selection.",
    )
    parser.add_argument(
        "--ratio-selection",
        choices=("score", "mape"),
        default="score",
        help="Choose the validation-eligible ratio candidate by aggregate score or MAPE.",
    )
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    source = args.source_run.resolve()
    checkpoint = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=False)
    source_metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    probe = json.loads(args.ratio_probe.resolve().read_text(encoding="utf-8"))
    if probe.get("dataset") != args.dataset or int(probe.get("seed")) != int(args.seed):
        raise RuntimeError("ratio probe dataset/seed mismatch")
    if probe.get("formal_test_target_sha256") != source_metrics.get("target_sha256"):
        raise RuntimeError("ratio probe target hash does not match source run")
    if probe.get("formal_source_metrics_sha256"):
        source_metrics_sha256 = hashlib.sha256(
            (source / "metrics.json").read_bytes()
        ).hexdigest()
        if probe["formal_source_metrics_sha256"] != source_metrics_sha256:
            raise RuntimeError("ratio probe source metrics hash does not match source run")
    selected_ratio = (
        probe.get("selected_by_mape")
        if args.ratio_selection == "mape"
        else probe.get("selected_by_score")
    )
    selected_ratio = selected_ratio or probe.get("selected_by_score") or probe.get("selected_by_mape")
    if selected_ratio is None:
        raise RuntimeError("ratio probe has no validation-selected candidate")
    ratio_family = str(selected_ratio["family"])
    if ratio_family not in {"horizon_node", "pred_bin_horizon"}:
        raise RuntimeError(f"unsupported ratio family: {ratio_family}")

    _, model, loaders, scaler, cfg = build_context(
        args.root.resolve(), args.run_root.resolve(), args.dataset, args.seed, device
    )
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
        )
        for loader in loaders[:2]
    ]
    feature_median = checkpoint["feature_median"]
    feature_scale = checkpoint["feature_scale"]
    for split in splits:
        split["feature_median"] = feature_median
        split["feature_scale"] = feature_scale

    train, val = splits
    # Validation-only selection of the formal CausalHistoryTemporalAdapter.
    train_context_median, train_context_scale = robust_statistics(train["context"])
    train_history_median, train_history_scale = robust_statistics(train["history"])
    train_values = (
        train["point"], train["target"],
        tocu.normalize_features(
            tocu.replace_point_features(train["raw_features"], train["point"]),
            feature_median, feature_scale,
        ),
        robust_normalize(train["context"], train_context_median, train_context_scale),
        robust_normalize(train["history"], train_history_median, train_history_scale),
        train["times"],
    )
    val_values = (
        val["point"], val["target"],
        tocu.normalize_features(
            tocu.replace_point_features(val["raw_features"], val["point"]),
            feature_median, feature_scale,
        ),
        robust_normalize(val["context"], train_context_median, train_context_scale),
        robust_normalize(val["history"], train_history_median, train_history_scale),
        val["times"],
    )
    formal_args = make_args(args, source_metrics)
    formal_args.dataset = args.dataset
    formal_args.seed = args.seed
    formal_args.batch_size = args.batch_size
    formal_args.grad_clip = args.grad_clip
    formal_args.history_blends = tuple(source_metrics.get("history_blends", [0.1, 0.25, 0.5, 0.75, 1.0]))
    formal_args.history_guard_tolerance = float(source_metrics.get("history_guard_tolerance", 0.001))
    formal_args.history_epochs = args.history_selection_epochs
    formal_args.history_patience = args.history_selection_patience
    selected_adapter, selection_history, selected_epoch, selected_mape, selection_baseline, selected_blend, selected_val = train_history_adapter(
        train_values[0], train_values[1], train_values[2], train_values[3], train_values[4], train_values[5],
        val_values[0], val_values[1], val_values[2], val_values[3], val_values[4], val_values[5],
        args.dataset, device, formal_args,
        point_cutoff=float(source_metrics.get("history_point_cutoff")),
    )
    if selected_epoch <= 0:
        raise RuntimeError("formal history selection produced no eligible epoch")

    # Refit causal statistics and the selected architecture on train+validation.
    fit_context_median, fit_context_scale = robust_statistics(
        torch.cat([train["context"], val["context"]], dim=0)
    )
    fit_history_median, fit_history_scale = robust_statistics(
        torch.cat([train["history"], val["history"]], dim=0)
    )
    fit_point = torch.cat([train["point"], val["point"]], dim=0)
    fit_target = torch.cat([train["target"], val["target"]], dim=0)
    fit_raw_features = torch.cat([train["raw_features"], val["raw_features"]], dim=0)
    fit_values = (
        fit_point,
        fit_target,
        tocu.normalize_features(
            tocu.replace_point_features(fit_raw_features, fit_point), feature_median, feature_scale
        ),
        robust_normalize(torch.cat([train["context"], val["context"]], dim=0), fit_context_median, fit_context_scale),
        robust_normalize(torch.cat([train["history"], val["history"]], dim=0), fit_history_median, fit_history_scale),
        torch.cat([train["times"], val["times"]], dim=0),
    )
    refit_history, refit_history_log = fit_history_fixed(
        fit_values, device, formal_args, selected_epoch
    )
    fit_cutoff = float(fit_point.max().item())
    train_refit = apply_history(refit_history, fit_values, device, args.batch_size, fit_cutoff)
    ratio_shrink = float(selected_ratio["shrink"])
    if ratio_family == "pred_bin_horizon":
        raw_ratio, ratio_edges = fit_pred_bin_horizon_ratio(
            train_refit,
            fit_target,
            args.dataset,
            int(selected_ratio["bin_count"]),
            float(selected_ratio["power"]),
            float(selected_ratio["cutoff"]),
            float(selected_ratio["ridge"]),
            device,
            gate=str(selected_ratio.get("gate", "below")),
        )
        prior = raw_ratio.mean(dim=1, keepdim=True)
        final_ratio = (1.0 - ratio_shrink) * raw_ratio + ratio_shrink * prior
        ratio_calibration = {
            "family": ratio_family,
            "gate": str(selected_ratio.get("gate", "below")),
            "bin_count": int(selected_ratio["bin_count"]),
            "cutoff": float(selected_ratio["cutoff"]),
            "relative_power": float(selected_ratio["power"]),
            "ridge": float(selected_ratio["ridge"]),
            "blend": float(selected_ratio["blend"]),
            "shrink": 0.0,
            "fit_shrink": ratio_shrink,
            "ratio": final_ratio.detach().cpu(),
            "edges": ratio_edges,
        }
    else:
        raw_ratio = tocu.fit_ratio_calibration(
            train_refit,
            fit_target,
            args.dataset,
            ratio_family,
            float(selected_ratio["power"]),
            float(selected_ratio["cutoff"]),
            1.0,
            gate=str(selected_ratio.get("gate", "below")),
        )
        prior = raw_ratio.mean(dim=1, keepdim=True)
        final_ratio = (1.0 - ratio_shrink) * raw_ratio + ratio_shrink * prior
        ratio_calibration = {
            "family": ratio_family,
            "gate": str(selected_ratio.get("gate", "below")),
            "cutoff": float(selected_ratio["cutoff"]),
            "relative_power": float(selected_ratio["power"]),
            "ridge": 1.0,
            "blend": float(selected_ratio["blend"]),
            "shrink": 0.0,
            "fit_shrink": ratio_shrink,
            "ratio": final_ratio.detach().cpu(),
        }
    train_final = tocu.apply_ratio_calibration(train_refit, ratio_calibration)
    fit_head_features = tocu.normalize_features(
        tocu.replace_point_features(fit_raw_features, train_final), feature_median, feature_scale
    )
    _, basis, pc_std, local_std, explained = fit_basis(
        fit_target - train_final, int(source_metrics.get("pc_rank", 3))
    )
    head_epochs = args.head_refit_epochs
    if head_epochs is None:
        head_epochs = int(source_metrics.get("head_epochs_completed", 1))
    head, head_history = fit_head_fixed(
        train_final, fit_target, fit_head_features, basis, pc_std, local_std,
        device, formal_args, head_epochs,
    )

    # Test sampling is the only phase that consumes the untouched test targets.
    seed_everything(args.seed + 26161)
    # Validate the point-path contract without touching the held-out test
    # loader. The test split is consumed only by the final evaluate() call.
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
        history_blend_alpha=1.0,
        history_point_cutoff=fit_cutoff,
        ratio_calibration=ratio_calibration,
    )
    all_targets = np.concatenate(targets, axis=0)[..., None]
    canonical = np.rint(all_targets).astype(np.float32, copy=False)
    canonical[canonical == 0] = 0.0
    ratio_hash = tocu.ratio_payload_sha256(ratio_calibration)
    ratio_selected_metadata = {
        key: value
        for key, value in ratio_calibration.items()
        if key not in {"ratio", "edges"}
    }
    if "edges" in ratio_calibration:
        ratio_selected_metadata["edges_shape"] = list(ratio_calibration["edges"].shape)
        ratio_selected_metadata["edges_sha256"] = sha256_array(
            ratio_calibration["edges"].numpy()
        )
    source_history = json.loads((source / "history.json").read_text(encoding="utf-8"))
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
        "training_mode": f"train_plus_validation_refit_{args.protocol_label}",
        "protocol_label": args.protocol_label,
        "backbone": "frozen_audited_source",
        "adapter_mode": source_metrics.get("adapter_mode", "horizon"),
        "adapter_blend_alpha": source_metrics.get("adapter_blend_alpha", 1.0),
        "selected_backbone_epoch": source_metrics.get("selected_backbone_epoch", 0),
        "selected_adapter_epoch": source_metrics.get("selected_adapter_epoch", 0),
        "selected_history_epoch": selected_epoch,
        "history_adapter_enabled": True,
        "history_adapter_epochs_completed": selected_epoch,
        "history_blend_alpha": 1.0,
        "history_point_cutoff": fit_cutoff,
        "history_context_refit": True,
        "history_selection_baseline_val": selection_baseline,
        "history_selection_val": selected_val,
        "history_selection_blend": selected_blend,
        "ratio_calibration_enabled": True,
        "ratio_calibration_mode": f"{ratio_family}_refit",
        "ratio_calibration_selected": ratio_selected_metadata,
        "ratio_calibration_sha256": ratio_hash,
        "ratio_calibration_source_probe": str(args.ratio_probe.resolve()),
        "ratio_probe_sha256": hashlib.sha256(args.ratio_probe.resolve().read_bytes()).hexdigest(),
        "ratio_calibration_selection_val": selected_ratio.get("val"),
        "head_epochs_completed": head_epochs,
        "selected_head_epoch": head_epochs,
        "head_refit_epochs_fixed_from_source_completed": True,
        "pc_rank": int(basis.shape[0]),
        "pc_explained_variance": explained,
        "mean_direction_count": int(checkpoint["mean_directions"].shape[0]),
        "conditioning_features": list(tocu.FEATURE_NAMES),
        "max_sample_mean_deviation": max_dev,
        "split": [0.6, 0.2, 0.2],
        "input_len": 12,
        "output_len": 12,
        "innovation_flags": source_metrics.get("innovation_flags", []),
        "optimization_flags": list(dict.fromkeys(source_metrics.get("optimization_flags", []) + [
            "train_plus_validation_history_refit",
            "validation_selected_horizon_node_ratio_refit",
            "train_plus_validation_uncertainty_refit",
        ])),
        "ratio_selection_criterion": args.ratio_selection,
        "metric_scales": source_metrics.get("metric_scales", {}),
        "mape_protocol": source_metrics.get("mape_protocol", "target_gt_1"),
        "crps_protocol": source_metrics.get("crps_protocol", "quantile_pinball_q05_to_q95_raw_target_div_sum_abs_target"),
        "mis_interval": source_metrics.get("mis_interval", [0.025, 0.975]),
        "target_sha256": sha256_array(canonical),
        "target_shape": list(canonical.shape),
        "test_read_policy": "test_loader_consumed_only_by_final_evaluate",
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    history_payload = {
        "backbone": source_history.get("backbone", []),
        "adapter": source_history.get("adapter", []),
        "history_selection": selection_history,
        "history": [
            {**row, "eligible": True} for row in refit_history_log
        ],
        "head": [
            {**row, "val_crps": None, "val_nll": None, "val_coverage": None}
            for row in head_history
        ],
    }
    (output / "history.json").write_text(json.dumps(history_payload, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {
            "model_state_dict": checkpoint["model_state_dict"],
            "adapter_state_dict": checkpoint["adapter_state_dict"],
            "head_state_dict": head.state_dict(),
            "feature_median": feature_median,
            "feature_scale": feature_scale,
            "mean_directions": checkpoint["mean_directions"],
            "basis": basis,
            "pc_std": pc_std,
            "local_std": local_std,
            "adapter_blend_alpha": source_metrics.get("adapter_blend_alpha", 1.0),
            "history_adapter_state_dict": refit_history.state_dict(),
            "history_context_median": fit_context_median,
            "history_context_scale": fit_context_scale,
            "history_feature_median": fit_history_median,
            "history_feature_scale": fit_history_scale,
            "history_blend_alpha": 1.0,
            "history_point_cutoff": fit_cutoff,
            "ratio_calibration": ratio_calibration,
        },
        output / "checkpoint.pt",
    )
    np.savez_compressed(
        output / "summary.npz",
        mean=np.concatenate(means)[..., None],
        lower=np.concatenate(lowers)[..., None],
        upper=np.concatenate(uppers)[..., None],
        target=np.concatenate(targets, axis=0)[..., None],
    )
    (output / "DONE").write_text("completed\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

