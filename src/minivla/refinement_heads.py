from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minivla.configuration_minivla import MiniVLAConfig


@dataclass
class RefinementConfig:
    """Configuration for post-SFT refinement heads.

    These heads are intentionally separate from the base policy. The usual
    training recipe freezes MiniVLA and trains these lightweight modules on
    validation, dry-run, recovery, or rollout-derived labels.
    """

    enable_action_probe: bool = True
    enable_verifier: bool = True
    enable_adaptive_horizon: bool = True
    enable_residual_recovery: bool = False
    probe_hidden_dim: int | None = None
    verifier_layers: int = 2
    residual_layers: int = 2
    horizon_choices: tuple[int, ...] = (1, 2, 4, 8, 16, 25)
    residual_scale: float = 0.25
    max_delta: float = 0.2
    action_min: float = -1.0
    action_max: float = 1.0
    saturation_eps: float = 1e-6
    extra: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.verifier_layers <= 0:
            raise ValueError("verifier_layers must be positive")
        if self.residual_layers <= 0:
            raise ValueError("residual_layers must be positive")
        if not self.horizon_choices:
            raise ValueError("horizon_choices cannot be empty")
        if any(choice <= 0 for choice in self.horizon_choices):
            raise ValueError("horizon choices must be positive")
        if self.residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        if self.max_delta < 0:
            raise ValueError("max_delta must be non-negative")
        if self.action_min > self.action_max:
            raise ValueError("action_min cannot exceed action_max")


def pad_action_chunk(actions: Tensor, action_dim: int) -> Tensor:
    if actions.shape[-1] >= action_dim:
        return actions[..., :action_dim]
    return F.pad(actions, (0, action_dim - actions.shape[-1]))


def action_chunk_features(
    actions: Tensor,
    previous_actions: Tensor | None = None,
    max_delta: float = 0.2,
    action_min: float = -1.0,
    action_max: float = 1.0,
    saturation_eps: float = 1e-6,
) -> Tensor:
    """Per-step action diagnostics used by probe and horizon heads."""

    if actions.ndim != 3:
        raise ValueError(f"actions must have shape (B,T,A), got {tuple(actions.shape)}")
    delta = torch.zeros_like(actions)
    delta[:, 1:] = actions[:, 1:] - actions[:, :-1]
    accel = torch.zeros_like(actions)
    accel[:, 2:] = actions[:, 2:] - 2.0 * actions[:, 1:-1] + actions[:, :-2]

    if previous_actions is not None:
        previous_actions = previous_actions.to(device=actions.device, dtype=actions.dtype)
        steps = min(previous_actions.shape[1], actions.shape[1])
        consistency = torch.zeros_like(actions)
        consistency[:, :steps] = actions[:, :steps] - previous_actions[:, :steps]
    else:
        consistency = torch.zeros_like(actions)

    magnitude = actions.norm(dim=-1, keepdim=True)
    step_delta_norm = delta.norm(dim=-1, keepdim=True)
    jerk_norm = accel.norm(dim=-1, keepdim=True)
    jump_flag = (delta.abs() > max_delta).float().mean(dim=-1, keepdim=True)
    saturation = ((actions <= action_min + saturation_eps) | (actions >= action_max - saturation_eps)).float().mean(
        dim=-1,
        keepdim=True,
    )
    return torch.cat(
        [
            actions,
            delta,
            accel,
            consistency,
            magnitude,
            step_delta_norm,
            jerk_norm,
            jump_flag,
            saturation,
        ],
        dim=-1,
    )


class ActionProbe(nn.Module):
    """Action-only failure probe inspired by action-space probe methods."""

    def __init__(self, policy_config: MiniVLAConfig, refinement_config: RefinementConfig | None = None):
        super().__init__()
        self.policy_config = policy_config
        self.refinement_config = refinement_config or RefinementConfig()
        hidden_dim = self.refinement_config.probe_hidden_dim or max(64, policy_config.hidden_dim // 2)
        feature_dim = policy_config.max_action_dim * 4 + 5
        self.input_norm = nn.LayerNorm(feature_dim)
        self.encoder = nn.LSTM(
            feature_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.step_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.chunk_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, actions: Tensor, previous_actions: Tensor | None = None) -> dict[str, Tensor]:
        actions = pad_action_chunk(actions, self.policy_config.max_action_dim)
        if previous_actions is not None:
            previous_actions = pad_action_chunk(previous_actions, self.policy_config.max_action_dim)
        features = action_chunk_features(
            actions,
            previous_actions=previous_actions,
            max_delta=self.refinement_config.max_delta,
            action_min=self.refinement_config.action_min,
            action_max=self.refinement_config.action_max,
            saturation_eps=self.refinement_config.saturation_eps,
        )
        encoded, _ = self.encoder(self.input_norm(features))
        pooled = encoded.mean(dim=1)
        failure_logit = self.chunk_head(pooled).squeeze(-1)
        step_failure_logits = self.step_head(encoded).squeeze(-1)
        return {
            "failure_logit": failure_logit,
            "failure_probability": torch.sigmoid(failure_logit),
            "step_failure_logits": step_failure_logits,
            "step_failure_probability": torch.sigmoid(step_failure_logits),
            "features": features,
        }


class ActionVerifierHead(nn.Module):
    """Scores a candidate action chunk conditioned on observation memory."""

    def __init__(self, policy_config: MiniVLAConfig, refinement_config: RefinementConfig | None = None):
        super().__init__()
        self.policy_config = policy_config
        self.refinement_config = refinement_config or RefinementConfig()
        self.action_in = nn.Linear(policy_config.max_action_dim, policy_config.hidden_dim)
        self.action_pos = nn.Parameter(torch.zeros(1, policy_config.chunk_size, policy_config.hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=policy_config.hidden_dim,
            nhead=policy_config.num_heads,
            dim_feedforward=int(policy_config.hidden_dim * policy_config.mlp_ratio),
            dropout=policy_config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.refinement_config.verifier_layers)
        self.norm = nn.LayerNorm(policy_config.hidden_dim)
        self.safety_head = nn.Sequential(
            nn.Linear(policy_config.hidden_dim, policy_config.hidden_dim),
            nn.GELU(),
            nn.Linear(policy_config.hidden_dim, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(policy_config.hidden_dim, policy_config.hidden_dim),
            nn.GELU(),
            nn.Linear(policy_config.hidden_dim, 1),
        )

    def forward(self, obs_tokens: Tensor, actions: Tensor) -> dict[str, Tensor]:
        actions = pad_action_chunk(actions, self.policy_config.max_action_dim).to(dtype=obs_tokens.dtype, device=obs_tokens.device)
        action_tokens = self.action_in(actions) + self.action_pos[:, : actions.shape[1]].to(dtype=obs_tokens.dtype)
        tokens = torch.cat([obs_tokens, action_tokens], dim=1)
        encoded = self.norm(self.encoder(tokens))
        action_encoded = encoded[:, -actions.shape[1] :]
        pooled = action_encoded.mean(dim=1)
        safety_logit = self.safety_head(pooled).squeeze(-1)
        advantage = self.advantage_head(pooled).squeeze(-1)
        return {
            "safety_logit": safety_logit,
            "safety_probability": torch.sigmoid(safety_logit),
            "advantage": advantage,
            "action_tokens": action_encoded,
        }


class AdaptiveHorizonHead(nn.Module):
    """Predicts how many action steps should be executed before replanning."""

    def __init__(self, policy_config: MiniVLAConfig, refinement_config: RefinementConfig | None = None):
        super().__init__()
        self.policy_config = policy_config
        self.refinement_config = refinement_config or RefinementConfig()
        feature_dim = policy_config.hidden_dim + policy_config.max_action_dim * 4 + 5
        hidden_dim = max(64, policy_config.hidden_dim // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.refinement_config.horizon_choices)),
        )

    def forward(self, obs_tokens: Tensor, actions: Tensor, previous_actions: Tensor | None = None) -> dict[str, Tensor]:
        actions = pad_action_chunk(actions, self.policy_config.max_action_dim).to(dtype=obs_tokens.dtype, device=obs_tokens.device)
        if previous_actions is not None:
            previous_actions = pad_action_chunk(previous_actions, self.policy_config.max_action_dim)
        features = action_chunk_features(
            actions,
            previous_actions=previous_actions,
            max_delta=self.refinement_config.max_delta,
            action_min=self.refinement_config.action_min,
            action_max=self.refinement_config.action_max,
            saturation_eps=self.refinement_config.saturation_eps,
        ).to(dtype=obs_tokens.dtype)
        pooled_features = features.mean(dim=1)
        pooled_obs = obs_tokens.mean(dim=1)
        logits = self.net(torch.cat([pooled_obs, pooled_features], dim=-1))
        choice_tensor = torch.tensor(self.refinement_config.horizon_choices, dtype=torch.float32, device=logits.device)
        probs = torch.softmax(logits, dim=-1)
        expected_horizon = (probs * choice_tensor[None, :]).sum(dim=-1)
        selected = torch.argmax(logits, dim=-1)
        horizons = torch.tensor(self.refinement_config.horizon_choices, dtype=torch.long, device=logits.device)[selected]
        return {
            "horizon_logits": logits,
            "horizon_probabilities": probs,
            "expected_horizon": expected_horizon,
            "horizon": horizons,
        }


class ResidualRecoveryPolicy(nn.Module):
    """Small residual policy that predicts delta actions without modifying MiniVLA."""

    def __init__(self, policy_config: MiniVLAConfig, refinement_config: RefinementConfig | None = None):
        super().__init__()
        self.policy_config = policy_config
        self.refinement_config = refinement_config or RefinementConfig()
        self.action_in = nn.Linear(policy_config.max_action_dim, policy_config.hidden_dim)
        self.action_pos = nn.Parameter(torch.zeros(1, policy_config.chunk_size, policy_config.hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=policy_config.hidden_dim,
            nhead=policy_config.num_heads,
            dim_feedforward=int(policy_config.hidden_dim * policy_config.mlp_ratio),
            dropout=policy_config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.refinement_config.residual_layers)
        self.norm = nn.LayerNorm(policy_config.hidden_dim)
        self.delta_out = nn.Linear(policy_config.hidden_dim, policy_config.max_action_dim)

    def forward(self, obs_tokens: Tensor, actions: Tensor) -> dict[str, Tensor]:
        actions = pad_action_chunk(actions, self.policy_config.max_action_dim).to(dtype=obs_tokens.dtype, device=obs_tokens.device)
        query = self.action_in(actions) + self.action_pos[:, : actions.shape[1]].to(dtype=obs_tokens.dtype)
        decoded = self.decoder(tgt=query, memory=obs_tokens)
        delta = torch.tanh(self.delta_out(self.norm(decoded))) * self.refinement_config.residual_scale
        refined = actions + delta
        return {
            "delta_action": delta[..., : self.policy_config.action_dim],
            "refined_actions": refined[..., : self.policy_config.action_dim],
        }


class PostSFTRefinementStack(nn.Module):
    """Composable post-SFT refinement stack for ablation experiments."""

    def __init__(self, policy_config: MiniVLAConfig, refinement_config: RefinementConfig | None = None):
        super().__init__()
        self.policy_config = policy_config
        self.refinement_config = refinement_config or RefinementConfig()
        self.action_probe = ActionProbe(policy_config, self.refinement_config) if self.refinement_config.enable_action_probe else None
        self.verifier = ActionVerifierHead(policy_config, self.refinement_config) if self.refinement_config.enable_verifier else None
        self.horizon_head = (
            AdaptiveHorizonHead(policy_config, self.refinement_config)
            if self.refinement_config.enable_adaptive_horizon
            else None
        )
        self.recovery_policy = (
            ResidualRecoveryPolicy(policy_config, self.refinement_config)
            if self.refinement_config.enable_residual_recovery
            else None
        )

    def forward(
        self,
        obs_tokens: Tensor,
        actions: Tensor,
        previous_actions: Tensor | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        out: dict[str, Tensor | dict[str, Tensor]] = {}
        if self.action_probe is not None:
            out["probe"] = self.action_probe(actions, previous_actions=previous_actions)
        if self.verifier is not None:
            out["verifier"] = self.verifier(obs_tokens, actions)
        if self.horizon_head is not None:
            out["horizon"] = self.horizon_head(obs_tokens, actions, previous_actions=previous_actions)
        if self.recovery_policy is not None:
            recovery = self.recovery_policy(obs_tokens, actions)
            out["recovery"] = recovery
            out["refined_actions"] = recovery["refined_actions"]
        else:
            out["refined_actions"] = actions[..., : self.policy_config.action_dim]
        return out

    def save_pretrained(
        self,
        output_dir: str | Path,
        base_checkpoint: str | None = None,
        assets: dict[str, Any] | None = None,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "refinement_model": self.state_dict(),
            "refinement_config": asdict(self.refinement_config),
            "base_checkpoint": base_checkpoint,
            "assets": dict(assets or {}),
        }
        torch.save(checkpoint, output_dir / "refinement.pt")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        policy_config: MiniVLAConfig,
        map_location: str | torch.device | None = None,
    ) -> "PostSFTRefinementStack":
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "refinement.pt"
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        config_dict = checkpoint.get("refinement_config", {})
        valid_keys = {field.name for field in fields(RefinementConfig)}
        refinement_config = RefinementConfig(**{key: value for key, value in config_dict.items() if key in valid_keys})
        stack = cls(policy_config, refinement_config)
        stack.load_state_dict(checkpoint["refinement_model"])
        return stack
