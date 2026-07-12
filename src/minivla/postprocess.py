from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class PostProcessConfig:
    action_dim: int | None = None
    action_min: float = -1.0
    action_max: float = 1.0
    max_delta: float | None = 0.2
    joint_min: float | None = None
    joint_max: float | None = None
    action_mode: str = "delta"
    ema_alpha: float = 0.35
    saturation_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.action_min > self.action_max:
            raise ValueError("action_min cannot exceed action_max")
        if self.max_delta is not None and self.max_delta < 0:
            raise ValueError("max_delta must be non-negative")
        if self.joint_min is not None and self.joint_max is not None and self.joint_min > self.joint_max:
            raise ValueError("joint_min cannot exceed joint_max")
        if self.action_mode not in {"delta", "absolute"}:
            raise ValueError("action_mode must be 'delta' or 'absolute'")
        if not 0.0 <= self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in [0, 1]")
        if self.saturation_eps < 0:
            raise ValueError("saturation_eps must be non-negative")


@dataclass
class PostProcessInfo:
    nonfinite_count: int = 0
    saturation_ratio: float = 0.0
    command_jump_ratio: float = 0.0
    max_abs_delta: float = 0.0
    joint_limit_projection: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActionPostProcessor:
    """Stateful safety and smoothing layer for policy actions.

    This module is intentionally model-agnostic: it can be used by checkpoint
    calibration, policy servers, or a robot-side client. The model predicts a
    command; this layer turns it into a bounded executable command.
    """

    def __init__(self, config: PostProcessConfig | None = None) -> None:
        self.config = config or PostProcessConfig()
        self.prev_action: Tensor | None = None
        self.ema_value: Tensor | None = None

    def reset(self) -> None:
        self.prev_action = None
        self.ema_value = None

    def select_action(self, actions: Tensor, batch_index: int = 0, step_index: int = 0) -> Tensor:
        if actions.ndim == 3:
            action = actions[batch_index, step_index]
        elif actions.ndim == 2:
            action = actions[batch_index]
        elif actions.ndim == 1:
            action = actions
        else:
            raise ValueError(f"Expected action tensor with 1-3 dims, got {tuple(actions.shape)}")
        if self.config.action_dim is not None:
            action = action[..., : self.config.action_dim]
        return action.clone()

    def unnormalize_action(self, action: Tensor, normalizer: Any | None = None) -> Tensor:
        if normalizer is None:
            return action
        if not hasattr(normalizer, "unnormalize_actions"):
            raise TypeError("normalizer must provide unnormalize_actions(actions)")
        original_shape = action.shape
        if action.ndim == 1:
            action_batch = action[None, None, :]
        elif action.ndim == 2:
            action_batch = action[:, None, :]
        elif action.ndim == 3:
            action_batch = action
        else:
            raise ValueError(f"Expected action tensor with 1-3 dims, got {tuple(action.shape)}")
        out = normalizer.unnormalize_actions(action_batch)
        if len(original_shape) == 1:
            return out[0, 0]
        if len(original_shape) == 2:
            return out[:, 0]
        return out

    def __call__(
        self,
        action: Tensor,
        current_joints: Tensor | None = None,
    ) -> tuple[Tensor, PostProcessInfo]:
        cfg = self.config
        info = PostProcessInfo()

        action = action.clone().to(dtype=torch.float32)
        finite = torch.isfinite(action)
        info.nonfinite_count = int((~finite).sum().item())
        if info.nonfinite_count:
            action = torch.nan_to_num(action, nan=0.0, posinf=cfg.action_max, neginf=cfg.action_min)

        action = action.clamp(cfg.action_min, cfg.action_max)
        near_min = action <= cfg.action_min + cfg.saturation_eps
        near_max = action >= cfg.action_max - cfg.saturation_eps
        info.saturation_ratio = float((near_min | near_max).float().mean().item())

        if cfg.max_delta is not None and self.prev_action is not None:
            delta = (action - self.prev_action).clamp(-cfg.max_delta, cfg.max_delta)
            unclipped_delta = action - self.prev_action
            info.command_jump_ratio = float((unclipped_delta.abs() > cfg.max_delta).float().mean().item())
            info.max_abs_delta = float(unclipped_delta.abs().max().item())
            action = self.prev_action + delta
        elif self.prev_action is not None:
            delta = action - self.prev_action
            info.max_abs_delta = float(delta.abs().max().item())

        if current_joints is not None and cfg.joint_min is not None and cfg.joint_max is not None:
            joint_dims = min(current_joints.numel(), action.numel())
            if cfg.action_mode == "delta":
                next_joints = current_joints[:joint_dims] + action[:joint_dims]
            else:
                next_joints = action[:joint_dims]
            projected = next_joints.clamp(cfg.joint_min, cfg.joint_max)
            info.joint_limit_projection = bool(torch.any(projected != next_joints).item())
            action = action.clone()
            if cfg.action_mode == "delta":
                action[:joint_dims] = projected - current_joints[:joint_dims]
            else:
                action[:joint_dims] = projected

        if self.ema_value is None:
            self.ema_value = action.clone()
        else:
            self.ema_value = cfg.ema_alpha * action + (1.0 - cfg.ema_alpha) * self.ema_value
        action = self.ema_value.clone()

        self.prev_action = action.clone()
        return action, info


class LatencyMonitor:
    def __init__(self) -> None:
        self.values_ms: list[float] = []

    def record(self, elapsed_s: float) -> float:
        value_ms = elapsed_s * 1000.0
        self.values_ms.append(value_ms)
        return value_ms

    def time_block(self) -> "_LatencyBlock":
        return _LatencyBlock(self)

    def summary(self) -> dict[str, float | int]:
        if not self.values_ms:
            return {"count": 0, "mean_ms": 0.0, "mean_hz": 0.0, "max_ms": 0.0}
        mean_ms = sum(self.values_ms) / len(self.values_ms)
        return {
            "count": len(self.values_ms),
            "mean_ms": mean_ms,
            "mean_hz": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
            "max_ms": max(self.values_ms),
        }


class _LatencyBlock:
    def __init__(self, monitor: LatencyMonitor) -> None:
        self.monitor = monitor
        self.start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> "_LatencyBlock":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed_ms = self.monitor.record(time.perf_counter() - self.start)


def action_smoothness(actions: Tensor) -> Tensor:
    if actions.shape[-2] < 2:
        return torch.zeros((), dtype=actions.dtype, device=actions.device)
    return (actions[..., 1:, :] - actions[..., :-1, :]).abs().mean()


def action_jerk(actions: Tensor) -> Tensor:
    if actions.shape[-2] < 3:
        return torch.zeros((), dtype=actions.dtype, device=actions.device)
    return (actions[..., 2:, :] - 2.0 * actions[..., 1:-1, :] + actions[..., :-2, :]).abs().mean()


def action_saturation_ratio(actions: Tensor, action_min: float, action_max: float, eps: float = 1e-6) -> Tensor:
    near_min = actions <= action_min + eps
    near_max = actions >= action_max - eps
    return (near_min | near_max).float().mean()


def command_jump_ratio(actions: Tensor, max_delta: float) -> Tensor:
    if actions.shape[-2] < 2:
        return torch.zeros((), dtype=actions.dtype, device=actions.device)
    deltas = actions[..., 1:, :] - actions[..., :-1, :]
    return (deltas.abs() > max_delta).float().mean()
