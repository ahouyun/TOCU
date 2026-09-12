"""Causal history-temporal point adapter for TOCU.

The adapter is deliberately separate from the external point backbone and
the low-rank residual adapter.  It starts at the identity, consumes
only the observed input window and future calendar phase, and is selected on
validation point metrics with a three-metric guard.
"""

from __future__ import annotations

import copy
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tocu.head import impedance_risk_from_flow, seed_everything


def _row_normalized_graph(adjacency: torch.Tensor, nodes: int, device, dtype) -> torch.Tensor:
    """Build a fixed row-normalized graph operator with an isolated-node fallback."""
    graph = torch.as_tensor(adjacency, device=device, dtype=dtype)
    if graph.ndim != 2 or tuple(graph.shape) != (nodes, nodes):
        raise ValueError(
            f"adjacency must be [{nodes}, {nodes}], got {tuple(graph.shape)}"
        )
    graph = graph.abs().clone()
    graph.fill_diagonal_(0.0)
    degree = graph.sum(-1, keepdim=True)
    normalized = graph / degree.clamp_min(1e-6)
    isolated = degree.squeeze(-1) <= 1e-6
    if bool(isolated.any()):
        identity = torch.eye(nodes, device=device, dtype=dtype)
        normalized = torch.where(isolated[:, None], identity, normalized)
    return normalized


def _graph_history_summary(log_flow: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
    """Summarize one normalized graph direction from the observed history."""
    neighbor = torch.einsum("ij,btj->bti", graph, log_flow)
    neighbor_recent = neighbor[:, -6:].movedim(1, 2)
    neighbor_diff = neighbor[:, 1:] - neighbor[:, :-1]
    neighbor_recent_diff = neighbor_diff[:, -3:].movedim(1, 2)
    neighbor_acceleration = (
        neighbor_diff[:, -1] - neighbor_diff[:, -2]
    ).unsqueeze(-1)
    local_neighbor_gap = (log_flow[:, -1] - neighbor[:, -1]).unsqueeze(-1)
    return torch.cat(
        [neighbor_recent, neighbor_recent_diff, neighbor_acceleration, local_neighbor_gap],
        dim=-1,
    )


def graph_history_features(
    flow: torch.Tensor,
    adjacency: torch.Tensor,
    mode: str = "outgoing",
) -> torch.Tensor:
    """Return causal neighbor-flow summaries from the fixed spatial graph.

    ``outgoing`` is the released compatibility path.  ``bidirectional`` adds
    the same summaries for the normalized transpose graph, allowing the
    adapter to distinguish incoming and outgoing traffic.  ``symmetric``
    uses a normalized union of both directions and keeps the original feature
    width.
    """
    if flow.ndim != 3 or flow.shape[1] < 6:
        raise ValueError(f"expected flow [B,T,N] with T>=6, got {tuple(flow.shape)}")
    if mode not in {"outgoing", "bidirectional", "symmetric"}:
        raise ValueError(f"unsupported graph history mode: {mode}")
    base = torch.as_tensor(adjacency, device=flow.device, dtype=flow.dtype)
    if mode == "symmetric":
        base = base + base.transpose(0, 1)
    graph = _row_normalized_graph(base, flow.shape[-1], flow.device, flow.dtype)
    log_flow = torch.log1p(flow.clamp_min(0.0))
    summaries = [_graph_history_summary(log_flow, graph)]
    if mode == "bidirectional":
        reverse = _row_normalized_graph(
            base.transpose(0, 1), flow.shape[-1], flow.device, flow.dtype
        )
        summaries.append(_graph_history_summary(log_flow, reverse))
    return torch.cat(summaries, dim=-1)


def history_features(
    flow: torch.Tensor,
    adjacency: torch.Tensor | None = None,
    graph_mode: str = "outgoing",
) -> torch.Tensor:
    """Return causal point-history features, optionally with graph context."""
    if flow.ndim != 3 or flow.shape[1] < 6:
        raise ValueError(f"expected flow [B,T,N] with T>=6, got {tuple(flow.shape)}")
    log_flow = torch.log1p(flow.clamp_min(0.0))
    recent = log_flow[:, -6:].movedim(1, 2)
    diff = log_flow[:, 1:] - log_flow[:, :-1]
    recent_diff = diff[:, -3:].movedim(1, 2)
    acceleration = (diff[:, -1] - diff[:, -2]).unsqueeze(-1)
    local = torch.cat([recent, recent_diff, acceleration], dim=-1)
    if adjacency is None:
        return local
    return torch.cat(
        [local, graph_history_features(flow, adjacency, mode=graph_mode)], dim=-1
    )


def future_time_features(
    raw_inputs: torch.Tensor, daily_len: int, horizon: int
) -> torch.Tensor:
    """Encode future time-of-day from the last observed calendar channel."""
    if raw_inputs.ndim != 4 or raw_inputs.shape[-1] < 2:
        raise ValueError(
            "history adapter requires flow and time-of-day input channels; "
            f"got {tuple(raw_inputs.shape)}"
        )
    last_tod = raw_inputs[:, -1, 0, 1].clamp(0.0, 1.0)
    ticks = last_tod[:, None] * float(daily_len) + torch.arange(
        1, horizon + 1, device=raw_inputs.device
    )[None, :]
    future_tod = (ticks % float(daily_len)) / float(daily_len)
    phase = 2.0 * np.pi * future_tod
    return torch.stack(
        [
            torch.sin(phase),
            torch.cos(phase),
            future_tod,
            last_tod[:, None].expand(-1, horizon),
        ],
        dim=-1,
    )


def causal_context_features(
    flow: torch.Tensor, congestion: torch.Tensor
) -> torch.Tensor:
    """Build per-node context using only the observed flow window."""
    if flow.ndim != 3 or congestion.shape != flow[:, -1].shape:
        raise ValueError(
            f"flow/congestion shapes are incompatible: {tuple(flow.shape)}, "
            f"{tuple(congestion.shape)}"
        )
    return torch.stack(
        [
            torch.log1p(flow[:, -1].clamp_min(0.0)),
            flow[:, -1] - flow[:, 0],
            torch.log1p(flow.mean(1).clamp_min(0.0)),
            torch.log1p(flow.std(1, unbiased=False).clamp_min(0.0)),
            congestion,
            congestion.mean(1, keepdim=True).expand_as(congestion),
        ],
        dim=-1,
    )


def robust_statistics(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit train-only median/MAD statistics for causal feature normalization."""
    flat = values.reshape(-1, values.shape[-1])
    median = flat.median(0).values
    scale = (flat - median).abs().median(0).values.mul(1.4826).clamp_min(1e-4)
    return median, scale


def robust_normalize(
    values: torch.Tensor, median: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return ((values - median) / scale).clamp(-6.0, 6.0)


def collect_causal_features(
    loader: Iterable,
    capacity: torch.Tensor,
    adjacency: torch.Tensor,
    device: torch.device,
    daily_len: int,
    horizon: int,
    include_graph_history: bool = False,
    graph_mode: str = "outgoing",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect context/history/time tensors in deterministic loader order."""
    contexts: list[torch.Tensor] = []
    histories: list[torch.Tensor] = []
    times: list[torch.Tensor] = []
    for batch in loader:
        raw_inputs = torch.as_tensor(batch["inputs"], device=device).clone()
        flow = raw_inputs[..., 0]
        _, congestion = impedance_risk_from_flow(flow, capacity, adjacency)
        contexts.append(causal_context_features(flow, congestion).cpu())
        histories.append(
            history_features(
                flow,
                adjacency if include_graph_history else None,
                graph_mode=graph_mode,
            ).cpu()
        )
        times.append(future_time_features(raw_inputs, daily_len, horizon).cpu())
    if not contexts:
        raise ValueError("loader produced no batches")
    return torch.cat(contexts), torch.cat(histories), torch.cat(times)


class CausalHistoryTemporalAdapter(nn.Module):
    """Bounded log-ratio residual adapter conditioned on causal context."""

    def __init__(
        self,
        global_count: int,
        context_count: int,
        history_count: int,
        time_count: int,
        horizon: int,
        nodes: int,
        hidden: int = 128,
        node_dim: int = 24,
        horizon_dim: int = 16,
        max_delta: float = 0.8,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.nodes = int(nodes)
        self.max_delta = float(max_delta)
        self.node_embedding = nn.Embedding(nodes, node_dim)
        self.horizon_embedding = nn.Embedding(horizon, horizon_dim)
        input_dim = (
            1
            + int(global_count)
            + int(context_count)
            + int(history_count)
            + int(time_count)
            + node_dim
            + horizon_dim
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.node_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.horizon_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("node_index", torch.arange(nodes))
        self.register_buffer("horizon_index", torch.arange(horizon))

    def forward(
        self,
        point: torch.Tensor,
        global_features: torch.Tensor,
        context: torch.Tensor,
        history: torch.Tensor,
        future_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if point.ndim != 3:
            raise ValueError(f"point must be [B,H,N], got {tuple(point.shape)}")
        batch, horizon, nodes = point.shape
        if horizon != self.horizon or nodes != self.nodes:
            raise ValueError(
                f"point shape {tuple(point.shape)} does not match adapter "
                f"[H={self.horizon}, N={self.nodes}]"
            )
        if context.shape[:2] != (batch, nodes) or history.shape[:2] != (batch, nodes):
            raise ValueError("context/history batch or node dimension mismatch")
        if future_time.shape[:2] != (batch, horizon):
            raise ValueError("future_time batch or horizon dimension mismatch")
        log_point = torch.log1p(point.clamp_min(0.0))
        global_part = global_features[:, None, None, :].expand(-1, horizon, nodes, -1)
        context_part = context[:, None, :, :].expand(-1, horizon, -1, -1)
        history_part = history[:, None, :, :].expand(-1, horizon, -1, -1)
        time_part = future_time[:, :, None, :].expand(-1, -1, nodes, -1)
        node_part = self.node_embedding(self.node_index)[None, None].expand(
            batch, horizon, -1, -1
        )
        horizon_part = self.horizon_embedding(self.horizon_index)[None, :, None].expand(
            batch, -1, nodes, -1
        )
        inputs = torch.cat(
            [
                log_point[..., None],
                global_part,
                context_part,
                history_part,
                time_part,
                node_part,
                horizon_part,
            ],
            dim=-1,
        )
        delta = self.max_delta * torch.tanh(self.net(inputs).squeeze(-1))
        # exp(log1p(point)+delta)-1 written this way keeps the zero-delta path
        # bitwise identical to the nonnegative point forecast.
        nonnegative_point = point.clamp_min(0.0)
        correction = (
            nonnegative_point * torch.exp(delta)
            + torch.expm1(delta)
            - nonnegative_point
        )
        corrected = point + correction
        return corrected, delta


def _point_metrics(point: torch.Tensor, target: torch.Tensor, dataset: str) -> dict[str, float]:
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


def _guarded(current: dict[str, float], baseline: dict[str, float], tolerance: float) -> bool:
    return all(
        current[key] <= baseline[key] * (1.0 + float(tolerance))
        for key in ("mae", "rmse", "mape")
    )


def _loss(
    corrected: torch.Tensor,
    target: torch.Tensor,
    delta: torch.Tensor,
    dataset: str,
    log_weight: float,
    relative_weight: float,
    relative_target_power: float,
    mae_weight: float,
    rmse_weight: float,
    delta_penalty: float,
    horizon_weight: float,
) -> torch.Tensor:
    threshold = 10.0 if dataset == "Seattle" else 1.0
    residual = corrected - target
    mask = target > threshold
    weights = torch.ones_like(target)
    if relative_target_power:
        reference = target.abs().mean().detach().clamp_min(threshold)
        weights = (
            reference / target.abs().clamp_min(threshold)
        ).pow(float(relative_target_power)).clamp(0.5, 4.0)
    if horizon_weight:
        horizon = torch.linspace(
            1.0,
            1.0 + float(horizon_weight),
            target.shape[1],
            device=target.device,
            dtype=target.dtype,
        )
        weights = weights * horizon.reshape(1, -1, 1)
    relative_values = residual.abs() / target.abs().clamp_min(1e-6)
    relative = (relative_values * weights)[mask].sum() / weights[mask].sum().clamp_min(1e-6)
    # The external backbone can expose a signed point forecast on some nodes.  The adapter
    # keeps that signed forecast unchanged at zero correction, but the
    # log-flow auxiliary term is only defined on the nonnegative flow domain.
    # Clamp only this auxiliary view; do not clamp ``corrected`` itself or the
    # identity path would silently change the point forecast.
    log_loss = F.smooth_l1_loss(
        torch.log1p(corrected.clamp_min(0.0)),
        torch.log1p(target.clamp_min(0.0)),
        beta=0.05,
    )
    scale = target.abs().mean().clamp_min(1.0)
    mae = residual.abs().mean() / scale
    rmse = residual.square().mean().sqrt() / scale
    return (
        float(log_weight) * log_loss
        + float(relative_weight) * relative
        + float(mae_weight) * mae
        + float(rmse_weight) * rmse
        + float(delta_penalty) * delta.square().mean()
    )


def apply_history_adapter(
    adapter: CausalHistoryTemporalAdapter,
    point: torch.Tensor,
    global_features: torch.Tensor,
    context: torch.Tensor,
    history: torch.Tensor,
    future_time: torch.Tensor,
    device: torch.device,
    batch_size: int,
    blend_alpha: float,
    point_cutoff: float | None = None,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    adapter.eval()
    with torch.no_grad():
        for start in range(0, point.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), point.shape[0])
            corrected, _ = adapter(
                point[start:stop].to(device),
                global_features[start:stop].to(device),
                context[start:stop].to(device),
                history[start:stop].to(device),
                future_time[start:stop].to(device),
            )
            base = point[start:stop].to(device)
            if point_cutoff is not None:
                gate = (base < float(point_cutoff)).to(base.dtype)
                corrected = base + gate * (corrected - base)
            outputs.append(
                (base + float(blend_alpha) * (corrected - base)).cpu()
            )
    return torch.cat(outputs)


def apply_history_support_gate(
    point: torch.Tensor, corrected: torch.Tensor, point_cutoff: float | None
) -> torch.Tensor:
    """Suppress history corrections outside the train-supported point range."""
    if point_cutoff is None:
        return corrected
    gate = (point < float(point_cutoff)).to(point.dtype)
    return point + gate * (corrected - point)


def train_history_adapter(
    train_point: torch.Tensor,
    train_target: torch.Tensor,
    train_global: torch.Tensor,
    train_context: torch.Tensor,
    train_history: torch.Tensor,
    train_time: torch.Tensor,
    val_point: torch.Tensor,
    val_target: torch.Tensor,
    val_global: torch.Tensor,
    val_context: torch.Tensor,
    val_history: torch.Tensor,
    val_time: torch.Tensor,
    dataset: str,
    device: torch.device,
    args,
    point_cutoff: float | None = None,
) -> tuple[CausalHistoryTemporalAdapter, list[dict], int, float, dict[str, float], float, dict[str, float]]:
    """Fit and validation-select the causal adapter without test access."""
    seed_everything(int(args.seed) + 9101)
    adapter = CausalHistoryTemporalAdapter(
        train_global.shape[1],
        train_context.shape[2],
        train_history.shape[2],
        train_time.shape[2],
        train_point.shape[1],
        train_point.shape[2],
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
    baseline = _point_metrics(val_point, val_target, dataset)
    best_state = copy.deepcopy(adapter.state_dict())
    best_epoch = 0
    best_blend = 0.0
    best_metrics = baseline
    selection_metric = str(getattr(args, "history_selection_metric", "mape"))
    if selection_metric not in {"score", "rmse", "mape"}:
        raise ValueError(f"unsupported history selection metric: {selection_metric}")
    best_value = (
        sum(baseline[key] / baseline[key] for key in ("mae", "rmse", "mape"))
        if selection_metric == "score"
        else baseline[selection_metric]
    )
    stale = 0
    history_log: list[dict] = []
    generator = torch.Generator().manual_seed(int(args.seed) + 9101)
    blends = tuple(float(x) for x in args.history_blends)
    for epoch in range(1, int(args.history_epochs) + 1):
        adapter.train()
        order = torch.randperm(train_point.shape[0], generator=generator)
        losses: list[float] = []
        for start in range(0, order.numel(), int(args.batch_size)):
            index = order[start : start + int(args.batch_size)]
            corrected, delta = adapter(
                train_point[index].to(device),
                train_global[index].to(device),
                train_context[index].to(device),
                train_history[index].to(device),
                train_time[index].to(device),
            )
            loss = _loss(
                corrected,
                train_target[index].to(device),
                delta,
                dataset,
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
        raw_val = apply_history_adapter(
            adapter,
            val_point,
            val_global,
            val_context,
            val_history,
            val_time,
            device,
            args.batch_size,
            1.0,
            point_cutoff=point_cutoff,
        )
        rows = []
        for blend in blends:
            mixed = val_point + blend * (raw_val - val_point)
            current = _point_metrics(mixed, val_target, dataset)
            rows.append(
                {
                    "blend": blend,
                    **current,
                    "selection_metric": selection_metric,
                    "eligible": _guarded(
                        current, baseline, args.history_guard_tolerance
                    ),
                }
            )
        eligible = [row for row in rows if row["eligible"]]
        if selection_metric == "score":
            selected = min(
                eligible,
                key=lambda row: (
                    sum(row[key] / baseline[key] for key in ("mae", "rmse", "mape")),
                    row["rmse"],
                    row["mape"],
                ),
            ) if eligible else None
        else:
            selected = min(
                eligible,
                key=lambda row: (row[selection_metric], row["mape"], row["rmse"]),
            ) if eligible else None
        history_log.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "selected": selected,
                "eligible_count": len(eligible),
            }
        )
        selected_value = None
        if selected is not None:
            selected_value = (
                sum(selected[key] / baseline[key] for key in ("mae", "rmse", "mape"))
                if selection_metric == "score"
                else selected[selection_metric]
            )
        if selected is not None and selected_value < best_value - 1e-5:
            best_state = copy.deepcopy(adapter.state_dict())
            best_epoch = epoch
            best_blend = float(selected["blend"])
            best_metrics = {key: float(selected[key]) for key in ("mae", "rmse", "mape")}
            best_value = float(selected_value)
            stale = 0
        else:
            stale += 1
            if stale >= int(args.history_patience):
                break
    adapter.load_state_dict(best_state)
    return (
        adapter,
        history_log,
        best_epoch,
        best_value,
        baseline,
        best_blend,
        best_metrics,
    )


__all__ = [
    "CausalHistoryTemporalAdapter",
    "apply_history_adapter",
    "apply_history_support_gate",
    "causal_context_features",
    "collect_causal_features",
    "future_time_features",
    "graph_history_features",
    "history_features",
    "robust_normalize",
    "robust_statistics",
    "train_history_adapter",
]
