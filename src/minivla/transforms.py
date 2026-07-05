from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from minivla.configuration_minivla import MiniVLAConfig
from minivla.constants import (
    ACTION,
    ACTION_DIM,
    FUTURE_IMAGE,
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    SUBGOAL_IMAGE,
)
from minivla.processor import MiniVLAProcessor


def pad_last_dim(tensor: Tensor, dim: int) -> Tensor:
    if tensor.shape[-1] >= dim:
        return tensor[..., :dim]
    return F.pad(tensor, (0, dim - tensor.shape[-1]))


def move_tensors(batch: Mapping[str, Any], device: torch.device | str) -> dict[str, Any]:
    out = dict(batch)
    for key, value in list(out.items()):
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
    return out


def repack_batch(batch: Mapping[str, Any], key_map: Mapping[str, str] | None = None) -> dict[str, Any]:
    out = dict(batch)
    if key_map is None:
        return out
    for source, target in key_map.items():
        if source in out and target not in out:
            out[target] = out[source]
    return out


def tokenize_batch(
    batch: Mapping[str, Any],
    processor: MiniVLAProcessor,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    return processor(dict(batch), device=device)


def convert_images(batch: Mapping[str, Any], image_size: tuple[int, int] | None = None) -> dict[str, Any]:
    out = dict(batch)
    for key, value in list(out.items()):
        if not torch.is_tensor(value):
            continue
        if key not in {OBS_IMAGE, SUBGOAL_IMAGE, FUTURE_IMAGE} and not key.startswith(f"{OBS_IMAGES}."):
            continue
        image = value
        if image.dtype == torch.uint8:
            image = image.float().div(255.0)
        elif not torch.is_floating_point(image):
            image = image.float()
        if image_size is not None:
            if image.ndim == 4:
                image = F.interpolate(image, size=image_size, mode="bilinear", align_corners=False)
            elif image.ndim == 5:
                bsz, steps, channels, height, width = image.shape
                image = image.reshape(bsz * steps, channels, height, width)
                image = F.interpolate(image, size=image_size, mode="bilinear", align_corners=False)
                image = image.reshape(bsz, steps, channels, *image_size)
        out[key] = image
    return out


def _to_tensor(value: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    try:
        return torch.as_tensor(value, device=device, dtype=dtype)
    except (TypeError, ValueError):
        return None


def _stat_tensor(stats: Mapping[str, Any], key: str, name: str, device: torch.device) -> Tensor | None:
    item = stats.get(key)
    if not isinstance(item, Mapping):
        return None
    value = item.get(name)
    if value is None and name == "std":
        min_value = _to_tensor(item.get("min"), device)
        max_value = _to_tensor(item.get("max"), device)
        if min_value is not None and max_value is not None:
            value = (max_value - min_value).clamp_min(1e-6)
    return _to_tensor(value, device)


class BatchNormalizer:
    """Normalize and unnormalize LeRobot-style state/action tensors."""

    def __init__(
        self,
        stats: Mapping[str, Any] | None,
        device: torch.device | str,
        normalize_state: bool = True,
        normalize_action: bool = True,
        eps: float = 1e-6,
    ) -> None:
        self.stats = dict(stats or {})
        self.device = torch.device(device)
        self.normalize_state = normalize_state
        self.normalize_action = normalize_action
        self.eps = eps

    def _moments(self, key: str) -> tuple[Tensor | None, Tensor | None]:
        return (
            _stat_tensor(self.stats, key, "mean", self.device),
            _stat_tensor(self.stats, key, "std", self.device),
        )

    def _match_shape(self, tensor: Tensor, mean: Tensor, std: Tensor) -> tuple[Tensor, Tensor]:
        while mean.ndim < tensor.ndim:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return mean.to(dtype=tensor.dtype), std.to(dtype=tensor.dtype).clamp_min(self.eps)

    def normalize_tensor(self, tensor: Tensor, key: str) -> Tensor:
        mean, std = self._moments(key)
        if mean is None or std is None:
            return tensor
        mean, std = self._match_shape(tensor, mean, std)
        return (tensor - mean) / std

    def unnormalize_tensor(self, tensor: Tensor, key: str) -> Tensor:
        mean, std = self._moments(key)
        if mean is None or std is None:
            return tensor
        mean, std = self._match_shape(tensor, mean, std)
        return tensor * std + mean

    def __call__(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(batch)
        if self.normalize_state and OBS_STATE in out and torch.is_tensor(out[OBS_STATE]):
            out[OBS_STATE] = self.normalize_tensor(out[OBS_STATE], OBS_STATE)
        if self.normalize_action and ACTION in out and torch.is_tensor(out[ACTION]):
            out[ACTION] = self.normalize_tensor(out[ACTION], ACTION)
        return out

    def unnormalize_actions(self, actions: Tensor) -> Tensor:
        if not self.normalize_action:
            return actions
        return self.unnormalize_tensor(actions, ACTION)


def prepare_batch(
    batch: Mapping[str, Any],
    config: MiniVLAConfig,
    processor: MiniVLAProcessor,
    normalizer: BatchNormalizer | None = None,
    device: torch.device | str | None = None,
    key_map: Mapping[str, str] | None = None,
    require_action: bool = True,
) -> dict[str, Any]:
    if device is None:
        device = config.device
    out = repack_batch(batch, key_map)
    if device is not None:
        out = move_tensors(out, device)
    out = convert_images(out, config.image_size)
    if normalizer is not None:
        out = normalizer(out)
    if ACTION in out and torch.is_tensor(out[ACTION]) and ACTION_DIM not in out:
        out[ACTION_DIM] = min(out[ACTION].shape[-1], config.max_action_dim)
    out = tokenize_batch(out, processor, device=device)

    if ACTION in out and torch.is_tensor(out[ACTION]) and out[ACTION].ndim == 2:
        out[ACTION] = out[ACTION].unsqueeze(1)
    if OBS_STATE in out and torch.is_tensor(out[OBS_STATE]):
        out[OBS_STATE] = pad_last_dim(out[OBS_STATE], config.max_state_dim)
    if ACTION in out and torch.is_tensor(out[ACTION]):
        out[ACTION] = pad_last_dim(out[ACTION], config.max_action_dim)

    required = [OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK]
    if require_action:
        required.append(ACTION)
    missing = [key for key in required if key not in out]
    if missing:
        raise KeyError(f"Prepared batch is missing required keys: {missing}")
    return out
