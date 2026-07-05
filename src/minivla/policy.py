from __future__ import annotations

import json
from dataclasses import asdict
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
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "policy.pt"
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

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str | Path,
        device: torch.device | str | None = None,
    ) -> "MiniVLAPolicyRunner":
        return cls.from_checkpoint(pretrained_path, device=device)

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_dict = asdict(self.policy.config)
        checkpoint = {
            "model": self.policy.state_dict(),
            "config": config_dict,
            "norm_stats": self.normalizer.stats,
            "dataset_stats": self.normalizer.stats,
            "assets": self.assets,
        }
        torch.save(checkpoint, output_dir / "policy.pt")
        with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(checkpoint["config"], handle, indent=2)
        with (output_dir / "assets.json").open("w", encoding="utf-8") as handle:
            json.dump(self.assets, handle, indent=2)
        with (output_dir / "model_card.md").open("w", encoding="utf-8") as handle:
            handle.write(self._model_card(config_dict))

    def _model_card(self, config_dict: dict[str, Any]) -> str:
        assets = self.assets
        dataset = assets.get("dataset_repo_id", "not recorded")
        image_keys = ", ".join(config_dict.get("image_keys", []))
        return f"""# MiniVLA Policy

This directory contains a MiniVLA policy checkpoint saved with
`MiniVLAPolicyRunner.save_pretrained`.

## Intended Use

MiniVLA is a compact pi0-style VLA stack for SO-101-scale experiments. This
checkpoint should be loaded with `MiniVLAPolicyRunner.from_pretrained` and used
with LeRobot-style observations.

## Configuration

- action_head: `{config_dict.get("action_head")}`
- image_keys: `{image_keys}`
- image_token_reduction: `{config_dict.get("image_token_reduction")}`
- max_image_tokens: `{config_dict.get("max_image_tokens")}`
- chunk_size: `{config_dict.get("chunk_size")}`
- n_action_steps: `{config_dict.get("n_action_steps")}`
- action_dim: `{config_dict.get("action_dim")}`
- use_temporal_ensemble: `{config_dict.get("use_temporal_ensemble")}`
- num_inference_steps: `{config_dict.get("num_inference_steps")}`

## Training Assets

- dataset_repo_id: `{dataset}`
- normalizer stats: stored in `policy.pt`
- config: `config.json`
- assets: `assets.json`

## Load

```python
from minivla import MiniVLAPolicyRunner

runner = MiniVLAPolicyRunner.from_pretrained("path/to/this/directory")
actions = runner.infer(observation)["actions"]
```

## Limitations

This model card is generated from checkpoint metadata. It does not report real
robot success rates or generalization claims unless those are added after
running controlled evaluation.
"""

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

    @torch.no_grad()
    def select_action(
        self,
        observation: dict[str, Any],
        unnormalize: bool = True,
    ) -> dict[str, torch.Tensor]:
        batch = prepare_batch(
            observation,
            self.policy.config,
            self.processor,
            self.normalizer,
            device=self.device,
            require_action=False,
        )
        action = self.policy.select_action(batch)
        if unnormalize:
            action = self.normalizer.unnormalize_actions(action[:, None, :])[:, 0]
        return {"action": action}
