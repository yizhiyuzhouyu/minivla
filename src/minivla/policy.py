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
from minivla.refinement_heads import PostSFTRefinementStack
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
        refinement_checkpoint: str | Path | None = None,
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
        runner = cls(policy, processor, normalizer, device=device, assets=assets)
        if refinement_checkpoint is not None:
            return MiniVLARefinedPolicyRunner.from_runner(runner, refinement_checkpoint)
        return runner

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str | Path,
        device: torch.device | str | None = None,
        refinement_checkpoint: str | Path | None = None,
    ) -> "MiniVLAPolicyRunner":
        return cls.from_checkpoint(pretrained_path, device=device, refinement_checkpoint=refinement_checkpoint)

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


class MiniVLARefinedPolicyRunner(MiniVLAPolicyRunner):
    """Inference wrapper that applies a trained post-SFT refinement stack."""

    def __init__(
        self,
        policy: MiniVLAPolicy,
        processor: MiniVLAProcessor,
        normalizer: BatchNormalizer,
        refinement: PostSFTRefinementStack,
        device: torch.device | str,
        assets: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(policy, processor, normalizer, device=device, assets=assets)
        self.refinement = refinement.to(self.device)
        self.refinement.eval()

    @classmethod
    def from_runner(
        cls,
        runner: MiniVLAPolicyRunner,
        refinement_checkpoint: str | Path,
    ) -> "MiniVLARefinedPolicyRunner":
        refinement = PostSFTRefinementStack.from_checkpoint(
            refinement_checkpoint,
            runner.policy.config,
            map_location=runner.device,
        ).to(runner.device)
        assets = dict(runner.assets)
        assets["refinement_checkpoint"] = str(refinement_checkpoint)
        return cls(
            runner.policy,
            runner.processor,
            runner.normalizer,
            refinement,
            device=runner.device,
            assets=assets,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        refinement_checkpoint: str | Path,
        device: torch.device | str | None = None,
    ) -> "MiniVLARefinedPolicyRunner":
        base_runner = MiniVLAPolicyRunner.from_checkpoint(checkpoint_path, device=device)
        return cls.from_runner(base_runner, refinement_checkpoint)

    def save_pretrained(self, output_dir: str | Path) -> None:
        super().save_pretrained(output_dir)
        output_dir = Path(output_dir)
        self.refinement.save_pretrained(
            output_dir,
            base_checkpoint=self.assets.get("base_checkpoint"),
            assets={"runner_assets": self.assets},
        )

    @torch.no_grad()
    def infer(
        self,
        observation: dict[str, Any],
        num_steps: int | None = None,
        unnormalize: bool = True,
        apply_residual: bool = True,
    ) -> dict[str, torch.Tensor]:
        batch = prepare_batch(
            observation,
            self.policy.config,
            self.processor,
            self.normalizer,
            device=self.device,
            require_action=False,
        )
        obs_tokens = self.policy.encode_observation_tokens(batch)
        actions = self.policy.predict_action_chunk(batch, num_steps=num_steps)
        refinement_out = self.refinement(obs_tokens.float(), actions.float())
        refined_actions = refinement_out.get("refined_actions", actions[..., : self.policy.config.action_dim])
        if apply_residual:
            actions = refined_actions
        else:
            actions = actions[..., : self.policy.config.action_dim]
        if unnormalize:
            actions = self.normalizer.unnormalize_actions(actions)
        result = {"actions": actions}
        if "probe" in refinement_out:
            result["failure_probability"] = refinement_out["probe"]["failure_probability"]
        if "verifier" in refinement_out:
            result["safety_probability"] = refinement_out["verifier"]["safety_probability"]
            result["advantage"] = refinement_out["verifier"]["advantage"]
        if "horizon" in refinement_out:
            result["horizon"] = refinement_out["horizon"]["horizon"]
            result["expected_horizon"] = refinement_out["horizon"]["expected_horizon"]
        return result

    @torch.no_grad()
    def select_action(
        self,
        observation: dict[str, Any],
        unnormalize: bool = True,
    ) -> dict[str, torch.Tensor]:
        result = self.infer(observation, unnormalize=unnormalize)
        action = result["actions"][:, 0]
        out = {"action": action}
        for key in ("failure_probability", "safety_probability", "horizon", "expected_horizon"):
            if key in result:
                out[key] = result[key]
        return out
