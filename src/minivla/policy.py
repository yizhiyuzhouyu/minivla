from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from minivla.configuration_minivla import MiniVLAConfig
from minivla.modeling_minivla import MiniVLAPolicy
from minivla.processor import MiniVLAProcessor
from minivla.transforms import BatchNormalizer, prepare_batch


def load_config(config_dict: dict[str, Any] | None) -> MiniVLAConfig:
    valid_keys = {field.name for field in fields(MiniVLAConfig)}
    return MiniVLAConfig(**{key: value for key, value in (config_dict or {}).items() if key in valid_keys})


class MiniVLAPolicyRunner:
    """Inference wrapper that mirrors the training transform path."""

    def __init__(
        self,
        policy: MiniVLAPolicy,
        processor: MiniVLAProcessor,
        normalizer: BatchNormalizer,
        device: torch.device | str,
        assets: dict[str, Any] | None = None,
    ) -> None:
        self.policy = policy
        self.processor = processor
        self.normalizer = normalizer
        self.device = torch.device(device)
        self.assets = dict(assets or {})
        self.policy.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | str | None = None,
    ) -> "MiniVLAPolicyRunner":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = load_config(checkpoint.get("config"))
        config.device = str(device)
        policy = MiniVLAPolicy(config).to(device)
        policy.load_compatible_state_dict(checkpoint["model"])
        processor = MiniVLAProcessor(config)
        assets = checkpoint.get("assets") or {}
        norm_stats = checkpoint.get("norm_stats", checkpoint.get("dataset_stats"))
        normalizer_assets = assets.get("normalizer", {}) if isinstance(assets.get("normalizer"), dict) else {}
        normalizer = BatchNormalizer(
            norm_stats,
            device=device,
            normalize_state=bool(normalizer_assets.get("normalize_state", assets.get("normalize_state", True))),
            normalize_action=bool(normalizer_assets.get("normalize_action", assets.get("normalize_action", True))),
        )
        return cls(policy, processor, normalizer, device=device, assets=assets)

    @torch.no_grad()
    def infer(self, observation: dict[str, Any], num_steps: int | None = None, unnormalize: bool = True) -> dict[str, torch.Tensor]:
        batch = prepare_batch(
            observation,
            self.policy.config,
            self.processor,
            self.normalizer,
            device=self.device,
            require_action=False,
        )
        actions = self.policy.predict_action_chunk(batch, num_steps=num_steps)
        if unnormalize:
            actions = self.normalizer.unnormalize_actions(actions)
        return {"actions": actions}
