from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Dataset

from minivla.configuration_minivla import MiniVLAConfig
from minivla.constants import ACTION
from minivla.refinement_heads import action_chunk_features, pad_action_chunk


LABEL_KEYS = ("success", "stable_grasp", "collision_free", "smooth_action")


@dataclass
class TrajectoryLabelWeights:
    success: float = 0.45
    stable_grasp: float = 0.2
    collision_free: float = 0.2
    smooth_action: float = 0.15
    human_score: float = 0.25


@dataclass
class RewardModelConfig:
    hidden_dim: int | None = None
    num_layers: int = 2
    dropout: float = 0.0
    label_weights: TrajectoryLabelWeights = field(default_factory=TrajectoryLabelWeights)

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.dropout < 0:
            raise ValueError("dropout must be non-negative")


@dataclass
class SACConfig:
    actor_hidden_dim: int | None = None
    critic_hidden_dim: int | None = None
    actor_layers: int = 2
    critic_layers: int = 2
    residual_scale: float = 0.25
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    gamma: float = 0.97
    tau: float = 0.005
    alpha: float = 0.01
    bc_weight: float = 0.05
    q_clip: float = 100.0

    def __post_init__(self) -> None:
        if self.actor_layers <= 0 or self.critic_layers <= 0:
            raise ValueError("actor_layers and critic_layers must be positive")
        if self.residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1]")
        if not 0 <= self.tau <= 1:
            raise ValueError("tau must be in [0, 1]")
        if self.alpha < 0 or self.bc_weight < 0:
            raise ValueError("alpha and bc_weight must be non-negative")


def _jsonl_paths(paths: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        item = Path(path)
        if item.is_dir():
            out.extend(sorted(item.glob("*.jsonl")))
        else:
            out.append(item)
    return out


def read_jsonl(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _jsonl_paths(paths):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    if isinstance(item, dict):
                        records.append(item)
    return records


def merge_label_files(records: list[dict[str, Any]], label_paths: Iterable[str | Path] | None) -> None:
    if label_paths is None:
        return
    labels_by_key: dict[str, dict[str, Any]] = {}
    for item in read_jsonl(label_paths):
        for key in ("trajectory_id", "episode_id", "episode"):
            if item.get(key) is not None:
                labels_by_key[str(item[key])] = item
    for record in records:
        for key in ("trajectory_id", "episode_id", "episode"):
            value = record.get(key)
            if value is not None and str(value) in labels_by_key:
                merged = dict(record.get("labels") or {})
                merged.update(labels_by_key[str(value)])
                record["labels"] = merged
                break


def _record_episode(record: dict[str, Any]) -> str:
    for key in ("trajectory_id", "episode_id", "episode", "rollout_id"):
        if record.get(key) is not None:
            return str(record[key])
    return "default"


def _record_cycle(record: dict[str, Any]) -> int:
    for key in ("cycle", "step", "timestep"):
        if record.get(key) is not None:
            return int(record[key])
    return 0


def _record_observation(record: dict[str, Any]) -> dict[str, Any]:
    observation = record.get("observation")
    if not isinstance(observation, dict):
        raise KeyError(
            "Rollout record is missing an 'observation' object. "
            "Run scripts/log_rollout.py with --save-observation."
        )
    return observation


def _record_action(record: dict[str, Any], prefer_postprocessed: bool = True) -> Any:
    if prefer_postprocessed and "postprocessed_action" in record:
        return record["postprocessed_action"]
    if "raw_action" in record:
        return record["raw_action"]
    if ACTION in record:
        return record[ACTION]
    raise KeyError("Rollout record is missing postprocessed_action/raw_action/action")


def labels_from_record(record: dict[str, Any], weights: TrajectoryLabelWeights | None = None) -> dict[str, float]:
    weights = weights or TrajectoryLabelWeights()
    source = dict(record.get("labels") or {})
    for key in LABEL_KEYS + ("human_score", "quality_score"):
        if key in record and key not in source:
            source[key] = record[key]

    labels: dict[str, float] = {}
    for key in LABEL_KEYS:
        value = source.get(key)
        if value is not None:
            labels[key] = float(value)
    human_score = source.get("human_score", source.get("quality_score"))
    if human_score is not None:
        labels["human_score"] = float(human_score)

    score = 0.0
    total = 0.0
    for key in LABEL_KEYS:
        if key in labels:
            weight = float(getattr(weights, key))
            score += weight * labels[key]
            total += weight
    if "human_score" in labels:
        weight = float(weights.human_score)
        score += weight * labels["human_score"]
        total += weight
    if total <= 0:
        raise KeyError(
            "Rollout record has no reward labels. Expected labels.success, "
            "labels.stable_grasp, labels.collision_free, labels.smooth_action, "
            "or labels.human_score."
        )
    labels["reward_target"] = score / total
    return labels


def _collate_value(values: list[Any]) -> Any:
    first = values[0]
    if torch.is_tensor(first):
        return torch.stack(values, dim=0)
    if isinstance(first, (float, int, bool)):
        return torch.as_tensor(values)
    if isinstance(first, str) or first is None:
        return values
    if isinstance(first, (list, tuple)):
        try:
            return torch.as_tensor(values)
        except (TypeError, ValueError):
            return values
    if isinstance(first, dict):
        keys = set().union(*(value.keys() for value in values if isinstance(value, dict)))
        return {key: _collate_value([value.get(key) if isinstance(value, dict) else None for value in values]) for key in keys}
    return values


def collate_rollout_windows(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {key: _collate_value([sample[key] for sample in samples]) for key in samples[0]}


class RolloutWindowDataset(Dataset):
    """Windowed SO-101 rollout records for reward-model and SAC training."""

    def __init__(
        self,
        rollout_paths: Iterable[str | Path],
        chunk_size: int,
        label_paths: Iterable[str | Path] | None = None,
        prefer_postprocessed_action: bool = True,
        require_labels: bool = True,
    ) -> None:
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        records = read_jsonl(rollout_paths)
        merge_label_files(records, label_paths)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault(_record_episode(record), []).append(record)
        self.windows: list[dict[str, Any]] = []
        for episode_id, episode_records in grouped.items():
            ordered = sorted(episode_records, key=_record_cycle)
            for index, record in enumerate(ordered):
                labels = labels_from_record(record) if require_labels else {}
                actions = []
                pad_mask = []
                for offset in range(self.chunk_size):
                    source_index = index + offset
                    if source_index < len(ordered):
                        actions.append(_record_action(ordered[source_index], prefer_postprocessed_action))
                        pad_mask.append(False)
                    else:
                        actions.append(actions[-1] if actions else _record_action(record, prefer_postprocessed_action))
                        pad_mask.append(True)
                next_record = ordered[min(index + 1, len(ordered) - 1)]
                self.windows.append(
                    {
                        "episode_id": episode_id,
                        "cycle": _record_cycle(record),
                        "observation": _record_observation(record),
                        "next_observation": _record_observation(next_record),
                        ACTION: torch.as_tensor(actions, dtype=torch.float32),
                        "action_is_pad": torch.as_tensor(pad_mask, dtype=torch.bool),
                        "done": float(index >= len(ordered) - 1),
                        **labels,
                    }
                )
        if not self.windows:
            raise RuntimeError("No rollout windows were built from the supplied rollout paths")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.windows[index]


def split_rollout_batch(batch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    current = dict(batch.get("observation") or {})
    next_batch = dict(batch.get("next_observation") or {})
    for key in (ACTION, "action_is_pad"):
        if key in batch:
            current[key] = batch[key]
    return current, next_batch


class TrajectoryRewardModel(nn.Module):
    """Scores observation-conditioned action chunks and predicts label heads."""

    def __init__(self, policy_config: MiniVLAConfig, reward_config: RewardModelConfig | None = None):
        super().__init__()
        self.policy_config = policy_config
        self.reward_config = reward_config or RewardModelConfig()
        hidden_dim = self.reward_config.hidden_dim or max(64, policy_config.hidden_dim // 2)
        feature_dim = policy_config.hidden_dim + policy_config.max_action_dim * 4 + 5
        layers: list[nn.Module] = [nn.LayerNorm(feature_dim)]
        in_dim = feature_dim
        for _ in range(self.reward_config.num_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.reward_config.dropout),
                ]
            )
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.score_head = nn.Linear(hidden_dim, 1)
        self.label_heads = nn.ModuleDict({key: nn.Linear(hidden_dim, 1) for key in LABEL_KEYS})

    def forward(self, obs_tokens: Tensor, actions: Tensor) -> dict[str, Tensor]:
        actions = pad_action_chunk(actions, self.policy_config.max_action_dim).to(device=obs_tokens.device, dtype=obs_tokens.dtype)
        features = action_chunk_features(actions).to(dtype=obs_tokens.dtype)
        x = torch.cat([obs_tokens.mean(dim=1), features.mean(dim=1)], dim=-1)
        hidden = self.trunk(x)
        score = self.score_head(hidden).squeeze(-1)
        labels = {key: self.label_heads[key](hidden).squeeze(-1) for key in self.label_heads}
        return {
            "reward_score": score,
            "reward": torch.sigmoid(score),
            **{f"{key}_logit": value for key, value in labels.items()},
        }

    def loss(self, outputs: dict[str, Tensor], batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        target = batch["reward_target"].to(device=outputs["reward"].device, dtype=outputs["reward"].dtype)
        loss = F.mse_loss(outputs["reward"], target)
        logs = {"reward_mse": float(loss.detach().cpu())}
        aux_losses = []
        for key in LABEL_KEYS:
            if key not in batch:
                continue
            label = batch[key].to(device=outputs["reward"].device, dtype=outputs["reward"].dtype)
            aux_losses.append(F.binary_cross_entropy_with_logits(outputs[f"{key}_logit"], label))
        if aux_losses:
            aux = torch.stack(aux_losses).mean()
            loss = loss + aux
            logs["label_bce"] = float(aux.detach().cpu())
        logs["loss"] = float(loss.detach().cpu())
        return loss, logs

    def save_pretrained(self, output_dir: str | Path, base_checkpoint: str | Path | None = None) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "reward_model": self.state_dict(),
                "reward_config": asdict(self.reward_config),
                "base_checkpoint": None if base_checkpoint is None else str(base_checkpoint),
            },
            output_dir / "reward_model.pt",
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        policy_config: MiniVLAConfig,
        map_location: str | torch.device | None = None,
    ) -> "TrajectoryRewardModel":
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "reward_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        config_dict = checkpoint.get("reward_config", {})
        valid = {field.name for field in fields(RewardModelConfig)}
        weights_dict = config_dict.get("label_weights")
        if isinstance(weights_dict, dict):
            config_dict = dict(config_dict)
            config_dict["label_weights"] = TrajectoryLabelWeights(**weights_dict)
        reward_config = RewardModelConfig(**{key: value for key, value in config_dict.items() if key in valid})
        model = cls(policy_config, reward_config)
        model.load_state_dict(checkpoint["reward_model"])
        return model


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, layers: int) -> nn.Sequential:
    modules: list[nn.Module] = [nn.LayerNorm(input_dim)]
    in_dim = input_dim
    for _ in range(layers):
        modules.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
        in_dim = hidden_dim
    modules.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*modules)


class ResidualSACActor(nn.Module):
    """Gaussian residual actor around the frozen MiniVLA action chunk."""

    def __init__(self, policy_config: MiniVLAConfig, sac_config: SACConfig | None = None):
        super().__init__()
        self.policy_config = policy_config
        self.sac_config = sac_config or SACConfig()
        hidden_dim = self.sac_config.actor_hidden_dim or max(64, policy_config.hidden_dim // 2)
        action_dim = policy_config.max_action_dim
        input_dim = policy_config.hidden_dim + action_dim * 4 + 5
        output_dim = policy_config.chunk_size * action_dim * 2
        self.net = _mlp(input_dim, hidden_dim, output_dim, self.sac_config.actor_layers)

    def forward(self, obs_tokens: Tensor, base_actions: Tensor) -> tuple[Tensor, Tensor]:
        base_actions = pad_action_chunk(base_actions, self.policy_config.max_action_dim).to(
            device=obs_tokens.device,
            dtype=obs_tokens.dtype,
        )
        features = action_chunk_features(base_actions).mean(dim=1).to(dtype=obs_tokens.dtype)
        out = self.net(torch.cat([obs_tokens.mean(dim=1), features], dim=-1))
        mean, log_std = out.chunk(2, dim=-1)
        mean = mean.view(base_actions.shape)
        log_std = log_std.view(base_actions.shape).clamp(self.sac_config.log_std_min, self.sac_config.log_std_max)
        return mean, log_std

    def sample(self, obs_tokens: Tensor, base_actions: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_std = self(obs_tokens, base_actions)
        if deterministic:
            z = mean
        else:
            z = mean + torch.randn_like(mean) * log_std.exp()
        residual = torch.tanh(z) * self.sac_config.residual_scale
        actions = pad_action_chunk(base_actions, self.policy_config.max_action_dim) + residual
        log_prob = _tanh_normal_log_prob(z, mean, log_std)
        return actions[..., : self.policy_config.action_dim], log_prob, residual[..., : self.policy_config.action_dim]


def _tanh_normal_log_prob(z: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    var = (2.0 * log_std).exp()
    log_prob = -0.5 * (((z - mean) ** 2) / var + 2.0 * log_std + torch.log(torch.tensor(2.0 * torch.pi, device=z.device)))
    correction = torch.log(1.0 - torch.tanh(z).pow(2) + 1e-6)
    return (log_prob - correction).flatten(start_dim=1).sum(dim=1)


class SACQNetwork(nn.Module):
    def __init__(self, policy_config: MiniVLAConfig, sac_config: SACConfig | None = None):
        super().__init__()
        self.policy_config = policy_config
        self.sac_config = sac_config or SACConfig()
        hidden_dim = self.sac_config.critic_hidden_dim or max(64, policy_config.hidden_dim // 2)
        feature_dim = policy_config.hidden_dim + policy_config.max_action_dim * 4 + 5
        self.net = _mlp(feature_dim, hidden_dim, 1, self.sac_config.critic_layers)

    def forward(self, obs_tokens: Tensor, actions: Tensor) -> Tensor:
        actions = pad_action_chunk(actions, self.policy_config.max_action_dim).to(device=obs_tokens.device, dtype=obs_tokens.dtype)
        features = action_chunk_features(actions).mean(dim=1).to(dtype=obs_tokens.dtype)
        return self.net(torch.cat([obs_tokens.mean(dim=1), features], dim=-1)).squeeze(-1)


class RHFSACAgent(nn.Module):
    def __init__(self, policy_config: MiniVLAConfig, sac_config: SACConfig | None = None):
        super().__init__()
        self.policy_config = policy_config
        self.sac_config = sac_config or SACConfig()
        self.actor = ResidualSACActor(policy_config, self.sac_config)
        self.q1 = SACQNetwork(policy_config, self.sac_config)
        self.q2 = SACQNetwork(policy_config, self.sac_config)
        self.target_q1 = SACQNetwork(policy_config, self.sac_config)
        self.target_q2 = SACQNetwork(policy_config, self.sac_config)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

    @torch.no_grad()
    def soft_update_targets(self) -> None:
        for target, source in ((self.target_q1, self.q1), (self.target_q2, self.q2)):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.mul_(1.0 - self.sac_config.tau).add_(param.data, alpha=self.sac_config.tau)

    def save_pretrained(
        self,
        output_dir: str | Path,
        base_checkpoint: str | Path | None = None,
        reward_checkpoint: str | Path | None = None,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sac_model": self.state_dict(),
                "sac_config": asdict(self.sac_config),
                "base_checkpoint": None if base_checkpoint is None else str(base_checkpoint),
                "reward_checkpoint": None if reward_checkpoint is None else str(reward_checkpoint),
            },
            output_dir / "sac.pt",
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        policy_config: MiniVLAConfig,
        map_location: str | torch.device | None = None,
    ) -> "RHFSACAgent":
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "sac.pt"
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        config_dict = checkpoint.get("sac_config", {})
        valid = {field.name for field in fields(SACConfig)}
        sac_config = SACConfig(**{key: value for key, value in config_dict.items() if key in valid})
        agent = cls(policy_config, sac_config)
        agent.load_state_dict(checkpoint["sac_model"])
        return agent
