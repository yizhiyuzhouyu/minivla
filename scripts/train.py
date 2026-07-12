from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minivla import MiniVLAConfig, MiniVLAProcessor, MiniVLAPolicy
from minivla.constants import (
    ACTION,
    ACTION_IS_PAD,
    EPISODE_QUALITY,
    EPISODE_SUCCESS,
    FUTURE_IMAGE,
    OBS_IMAGES,
    OBS_STATE,
    SUBGOAL_IMAGE,
    SUBTASK_LABEL,
)
from minivla.postprocess import action_jerk, action_smoothness
from minivla.splits import load_episode_split
from minivla.transforms import BatchNormalizer, prepare_batch


LOG_FIELDS = [
    "type",
    "step",
    "epoch",
    "loss",
    "lr",
    "sample_loss_mean",
    "sample_loss_max",
    "loss_clip_fraction",
    "sample_weight_mean",
    "sample_weight_max",
    "fm_action_smoothness_loss",
    "future_latent_loss",
    "fm_time_mean",
    "fm_noise_std",
    "val_mean_fm_loss",
    "val_mean_sampled_action_mse",
    "val_mean_sampled_action_mse_normalized",
    "val_mean_sampled_action_mse_original",
    "val_mean_action_smoothness",
    "val_mean_action_jerk",
    "val_latency_ms",
    "val_selection_score",
]


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _collate_value(values: list[Any]) -> Any:
    first = values[0]
    if torch.is_tensor(first):
        return torch.stack(values, dim=0)
    if isinstance(first, (float, int, bool)):
        return torch.as_tensor(values)
    if isinstance(first, str):
        return values
    if isinstance(first, dict):
        return {key: _collate_value([value[key] for value in values]) for key in first}
    return values


def collate_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    keys = samples[0].keys()
    return {key: _collate_value([sample[key] for sample in samples]) for key in keys}


def build_delta_timestamps(
    chunk_size: int,
    n_obs_steps: int,
    fps: float,
    image_keys: list[str] | tuple[str, ...],
) -> dict[str, list[float]]:
    delta_timestamps = {
        ACTION: [i / fps for i in range(chunk_size)],
    }
    if n_obs_steps > 1:
        obs_offsets = [-(n_obs_steps - 1 - i) / fps for i in range(n_obs_steps)]
        delta_timestamps[OBS_STATE] = obs_offsets
        for key in image_keys:
            if key.startswith(f"{OBS_IMAGES}.") or key == "observation.image":
                delta_timestamps[key] = obs_offsets
    return delta_timestamps


def load_lerobot_dataset(
    args: argparse.Namespace,
    split_json: str | None = None,
    split_name: str | None = None,
    episodes: list[int] | None = None,
):
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise ImportError(
                "Could not import LeRobotDataset. Install LeRobot or pass a compatible import path."
            ) from exc

    selected_episodes = args.episodes if episodes is None else episodes
    split_episodes = load_episode_split(split_json or args.split_json, split_name or args.split_name)
    if split_episodes is not None:
        selected_episodes = split_episodes
    kwargs: dict[str, Any] = {}
    if args.dataset_root is not None:
        kwargs["root"] = args.dataset_root
    if selected_episodes is not None:
        kwargs["episodes"] = selected_episodes

    if args.delta_timestamps:
        kwargs["delta_timestamps"] = build_delta_timestamps(
            args.chunk_size,
            args.n_obs_steps,
            args.fps,
            args.image_keys,
        )

    return LeRobotDataset(args.dataset_repo_id, **kwargs)


def dataset_stats(dataset: Any) -> dict[str, Any]:
    meta = getattr(dataset, "meta", None)
    for source in (dataset, meta):
        stats = getattr(source, "stats", None)
        if isinstance(stats, dict):
            return stats
    return {}


def config_from_args(args: argparse.Namespace, checkpoint: dict[str, Any] | None) -> MiniVLAConfig:
    config_keys = {field.name for field in fields(MiniVLAConfig)}
    config_dict: dict[str, Any] = {}
    if checkpoint is not None and args.use_checkpoint_config:
        saved_config = checkpoint.get("config")
        if isinstance(saved_config, dict):
            config_dict.update({key: value for key, value in saved_config.items() if key in config_keys})

    overrides = {
        "device": args.device,
        "dtype": args.dtype,
        "use_hf_vision_encoder": args.use_hf_vision_encoder,
        "freeze_vision_encoder": args.freeze_vision_encoder,
        "vision_model_name": args.vision_model_name,
        "tokenizer_name": args.tokenizer_name,
        "image_keys": tuple(args.image_keys),
        "image_size": tuple(args.image_size),
        "patch_size": args.patch_size,
        "image_token_reduction": args.image_token_reduction,
        "visual_resampler_layers": args.visual_resampler_layers,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_dit_layers": args.num_dit_layers,
        "action_head": args.action_head,
        "n_obs_steps": args.n_obs_steps,
        "chunk_size": args.chunk_size,
        "n_action_steps": args.n_action_steps,
        "max_state_dim": args.max_state_dim,
        "max_action_dim": args.max_action_dim,
        "action_dim": args.action_dim,
        "use_temporal_ensemble": args.use_temporal_ensemble,
        "temporal_ensemble_max_chunks": args.temporal_ensemble_max_chunks,
        "temporal_ensemble_decay": args.temporal_ensemble_decay,
        "use_episode_metadata": args.use_episode_metadata,
        "future_latent_loss_weight": args.future_latent_loss_weight,
        "text_vocab_size": args.text_vocab_size,
        "tokenizer_max_length": args.tokenizer_max_length,
        "num_inference_steps": args.num_inference_steps,
        "max_image_tokens": args.max_image_tokens,
        "fm_action_smoothness_loss_weight": args.fm_action_smoothness_loss_weight,
    }
    for key, value in overrides.items():
        if value is not None:
            config_dict[key] = value
    return MiniVLAConfig(**config_dict)


def save_checkpoint(
    path: Path,
    policy: MiniVLAPolicy,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    config: MiniVLAConfig,
    stats: dict[str, Any],
    assets: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "config": asdict(config),
            "norm_stats": stats,
            "dataset_stats": stats,
            "assets": assets,
        },
        path,
    )


def load_checkpoint(path: str | None, map_location: str | torch.device) -> dict[str, Any] | None:
    if path is None:
        return None
    return torch.load(path, map_location=map_location)


def load_yaml_defaults(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for --config. Install project dependencies first.") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping, got {type(data).__name__}")
    return {str(key).replace("-", "_"): value for key, value in data.items()}


def apply_parser_defaults(parser: argparse.ArgumentParser, defaults: dict[str, Any]) -> None:
    if not defaults:
        return
    valid_dests = {action.dest for action in parser._actions}
    unknown = sorted(key for key in defaults if key not in valid_dests)
    if unknown:
        raise ValueError(f"Unknown config keys in YAML: {unknown}")
    parser.set_defaults(**defaults)


def _as_sample_list(value: Any, batch_size: int) -> list[Any]:
    if value is None:
        return [None] * batch_size
    if torch.is_tensor(value):
        if value.ndim == 0:
            item = value.item()
            return [item] * batch_size
        out = []
        for index in range(batch_size):
            if index >= value.shape[0]:
                out.append(None)
                continue
            item = value[index]
            out.append(item.item() if item.numel() == 1 else item.detach().cpu().tolist())
        return out
    if isinstance(value, list):
        return [value[index] if index < len(value) else None for index in range(batch_size)]
    return [value] * batch_size


def _episode_keys(raw_batch: dict[str, Any], batch_size: int) -> list[str | None]:
    for key in ("episode_index", "episode.id", "episode", "episode_id"):
        if key in raw_batch:
            return [None if item is None else str(item) for item in _as_sample_list(raw_batch[key], batch_size)]
    return [None] * batch_size


def load_failure_episode_weights(path: str | None, upweight: float) -> dict[str, float]:
    if path is None:
        return {}
    weights: dict[str, float] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            episode = item.get("episode_index", item.get("episode.id", item.get("episode", item.get("episode_id"))))
            if episode is None:
                continue
            weights[str(episode)] = max(weights.get(str(episode), 1.0), upweight)
    return weights


def build_sample_weights(
    raw_batch: dict[str, Any],
    batch: dict[str, Any],
    args: argparse.Namespace,
    failure_episode_weights: dict[str, float],
    device: torch.device,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if not args.use_quality_weights and not failure_episode_weights and args.subtask_balance_weight <= 0:
        return None, {}

    batch_size = int(batch[ACTION].shape[0])
    weights = torch.ones(batch_size, dtype=torch.float32, device=device)

    if args.use_quality_weights:
        if EPISODE_SUCCESS in batch:
            success = batch[EPISODE_SUCCESS].to(device=device, dtype=torch.float32).flatten()[:batch_size]
            weights = weights * torch.where(
                success > 0.5,
                torch.full_like(weights, args.success_weight),
                torch.full_like(weights, args.failure_weight),
            )
        if EPISODE_QUALITY in batch:
            quality = batch[EPISODE_QUALITY].to(device=device, dtype=torch.float32).flatten()[:batch_size].clamp(0.0, 1.0)
            weights = weights * (1.0 + args.quality_weight_scale * quality)

    if failure_episode_weights:
        episode_weights = []
        for episode in _episode_keys(raw_batch, batch_size):
            episode_weights.append(failure_episode_weights.get(str(episode), 1.0) if episode is not None else 1.0)
        weights = weights * torch.tensor(episode_weights, dtype=torch.float32, device=device)

    if args.subtask_balance_weight > 0 and SUBTASK_LABEL in raw_batch:
        labels = _as_sample_list(raw_batch[SUBTASK_LABEL], batch_size)
        counts: dict[str, int] = {}
        for label in labels:
            if label is not None:
                counts[str(label)] = counts.get(str(label), 0) + 1
        if counts:
            inv = torch.tensor(
                [1.0 / counts.get(str(label), 1) if label is not None else 1.0 for label in labels],
                dtype=torch.float32,
                device=device,
            )
            inv = inv / inv.mean().clamp_min(1e-6)
            weights = weights * ((1.0 - args.subtask_balance_weight) + args.subtask_balance_weight * inv)

    weights = weights.clamp(args.min_sample_weight, args.max_sample_weight)
    return weights, {
        "sample_weight_mean": float(weights.mean().detach().cpu()),
        "sample_weight_max": float(weights.max().detach().cpu()),
    }


def per_sample_sft_loss(
    loss_tensor: torch.Tensor,
    action_pad_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_tensor = loss_tensor.float()
    sample_valid: torch.Tensor | None = None
    if action_pad_mask is not None and loss_tensor.ndim >= 3:
        if action_pad_mask.ndim == 1:
            action_pad_mask = action_pad_mask[:, None].expand(-1, loss_tensor.shape[1])
        if action_pad_mask.shape != loss_tensor.shape[:2]:
            raise ValueError(
                "action_pad_mask must have shape "
                f"{tuple(loss_tensor.shape[:2])}, got {tuple(action_pad_mask.shape)}"
            )
        valid = (~action_pad_mask.bool()).to(dtype=loss_tensor.dtype, device=loss_tensor.device)
        dims_per_step = math.prod(loss_tensor.shape[2:])
        sample_loss = loss_tensor.flatten(start_dim=1).sum(dim=1) / (valid.sum(dim=1).clamp_min(1.0) * dims_per_step)
        sample_valid = valid.sum(dim=1) > 0
    else:
        sample_loss = loss_tensor.flatten(start_dim=1).mean(dim=1)
        sample_valid = torch.ones_like(sample_loss, dtype=torch.bool)
    return sample_loss, {
        "valid_sample_fraction": float(sample_valid.float().mean().detach().cpu()) if sample_valid is not None else 1.0,
    }


def reduce_sft_loss(
    loss_tensor: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    loss_clip: float = 0.0,
    action_pad_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    sample_loss, sample_logs = per_sample_sft_loss(loss_tensor, action_pad_mask=action_pad_mask)
    sample_valid = torch.isfinite(sample_loss)
    if action_pad_mask is not None and loss_tensor.ndim >= 3:
        if action_pad_mask.ndim == 1:
            action_pad_mask = action_pad_mask[:, None].expand(-1, loss_tensor.shape[1])
        valid_steps = (~action_pad_mask.bool()).to(device=loss_tensor.device).sum(dim=1)
        sample_valid = sample_valid & (valid_steps > 0)
    unclipped = sample_loss
    clipped_fraction = 0.0
    if loss_clip is not None and loss_clip > 0:
        clipped_fraction = float((sample_loss > loss_clip).float().mean().detach().cpu())
        sample_loss = sample_loss.clamp(max=loss_clip)
    if sample_weights is not None:
        effective_weights = sample_weights.to(device=sample_loss.device, dtype=sample_loss.dtype)
        if sample_valid is not None:
            effective_weights = effective_weights * sample_valid.to(dtype=effective_weights.dtype)
        loss = (sample_loss * effective_weights).sum() / effective_weights.sum().clamp_min(1e-6)
    else:
        if sample_valid is not None and sample_valid.any():
            loss = sample_loss[sample_valid].mean()
        else:
            loss = sample_loss.mean()
    return loss, {
        "sample_loss_mean": float(unclipped.mean().detach().cpu()),
        "sample_loss_max": float(unclipped.max().detach().cpu()),
        "loss_clip_fraction": clipped_fraction,
        **sample_logs,
    }


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def write_csv(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(record)


@torch.no_grad()
def evaluate_policy(
    policy: MiniVLAPolicy,
    dataloader: DataLoader,
    processor: MiniVLAProcessor,
    normalizer: BatchNormalizer,
    config: MiniVLAConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    was_training = policy.training
    policy.eval()
    losses: list[float] = []
    mses: list[float] = []
    original_mses: list[float] = []
    smoothness_values: list[float] = []
    jerk_values: list[float] = []
    latencies_ms: list[float] = []

    for batch_index, raw_batch in enumerate(dataloader):
        if batch_index >= args.eval_max_batches:
            break
        batch = prepare_batch(raw_batch, config, processor, normalizer, device=device, require_action=True)
        loss, _ = policy(batch)
        losses.append(float(loss.detach().cpu()))

        obs_batch = {key: value for key, value in batch.items() if key != ACTION}
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        pred = policy.predict_action_chunk(obs_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        target = batch[ACTION][..., : pred.shape[-1]]
        steps = min(pred.shape[1], target.shape[1])
        pred = pred[:, :steps]
        target = target[:, :steps]
        mse_normalized = torch.mean((pred - target) ** 2)
        pred_original = normalizer.unnormalize_actions(pred)
        target_original = normalizer.unnormalize_actions(target)
        mse_original = torch.mean((pred_original - target_original) ** 2)
        mses.append(float(mse_normalized.detach().cpu()))
        smoothness_values.append(float(action_smoothness(pred).detach().cpu()))
        jerk_values.append(float(action_jerk(pred).detach().cpu()))
        original_mses.append(float(mse_original.detach().cpu()))

    if was_training:
        policy.train()
    if not losses:
        raise RuntimeError("Validation dataloader produced no batches")
    metrics = {
        "mean_fm_loss": sum(losses) / len(losses),
        "mean_sampled_action_mse": sum(mses) / len(mses),
        "mean_sampled_action_mse_normalized": sum(mses) / len(mses),
        "mean_sampled_action_mse_original": sum(original_mses) / len(original_mses),
        "mean_action_smoothness": sum(smoothness_values) / len(smoothness_values),
        "mean_action_jerk": sum(jerk_values) / len(jerk_values),
        "latency_ms": sum(latencies_ms) / len(latencies_ms),
    }
    metrics["selection_score"] = (
        metrics["mean_fm_loss"]
        + metrics["mean_sampled_action_mse"]
        + 0.1 * metrics["mean_action_smoothness"]
        + 0.1 * metrics["mean_action_jerk"]
        + 0.001 * metrics["latency_ms"]
    )
    return metrics


def build_parser(config_defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MiniVLA on a LeRobot dataset.")
    parser.add_argument("--config", default=None, help="Optional YAML config. CLI flags override config values.")
    parser.add_argument("--dataset-repo-id", default=None, help="LeRobot dataset repo id or local dataset id.")
    parser.add_argument("--dataset-root", default=None, help="Optional local dataset root.")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Optional episode ids to train on.")
    parser.add_argument("--split-json", default=None, help="Optional JSON file containing episode ids.")
    parser.add_argument("--split-name", default=None, help="Optional split key inside --split-json.")
    parser.add_argument("--val-split-json", default=None, help="Optional validation split JSON for train-time evaluation.")
    parser.add_argument("--val-split-name", default=None, help="Optional split key inside --val-split-json.")
    parser.add_argument("--output-dir", default="outputs/minivla", help="Directory for checkpoints.")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--use-checkpoint-config", action="store_true", help="Start from the checkpoint config before CLI overrides.")

    parser.add_argument("--device", default=_auto_device())
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=0, help="Evaluate on --val-split-json every N steps.")
    parser.add_argument("--eval-max-batches", type=int, default=20)
    parser.add_argument("--best-metric", default="selection_score")
    parser.add_argument("--train-log-jsonl", default=None)
    parser.add_argument("--train-log-csv", default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--normalize-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--delta-timestamps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fps", type=float, default=30.0)

    parser.add_argument("--use-hf-vision-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze-vision-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vision-model-name", default=None)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--image-keys", nargs="+", default=["observation.images.front"])
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--image-token-reduction", default="adaptive_pool", choices=["adaptive_pool", "resampler"])
    parser.add_argument("--visual-resampler-layers", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-dit-layers", type=int, default=4)
    parser.add_argument("--action-head", default="flow_matching", choices=["flow_matching", "mlp", "query"])
    parser.add_argument("--n-obs-steps", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=50)
    parser.add_argument("--max-state-dim", type=int, default=32)
    parser.add_argument("--max-action-dim", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--use-temporal-ensemble", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temporal-ensemble-max-chunks", type=int, default=8)
    parser.add_argument("--temporal-ensemble-decay", type=float, default=0.5)
    parser.add_argument("--use-episode-metadata", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--future-latent-loss-weight", type=float, default=0.0)
    parser.add_argument("--text-vocab-size", type=int, default=49152)
    parser.add_argument("--tokenizer-max-length", type=int, default=48)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--max-image-tokens", type=int, default=None)
    parser.add_argument("--fm-action-smoothness-loss-weight", type=float, default=0.0)
    parser.add_argument("--loss-clip", type=float, default=0.0, help="Clip per-sample SFT loss before weighting; 0 disables.")
    parser.add_argument("--use-quality-weights", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--success-weight", type=float, default=1.25)
    parser.add_argument("--failure-weight", type=float, default=0.75)
    parser.add_argument("--quality-weight-scale", type=float, default=0.5)
    parser.add_argument("--subtask-balance-weight", type=float, default=0.0)
    parser.add_argument("--failure-manifest", default=None, help="failure_cases.jsonl from post-SFT selection/replay.")
    parser.add_argument("--failure-upweight", type=float, default=2.0)
    parser.add_argument("--min-sample-weight", type=float, default=0.1)
    parser.add_argument("--max-sample-weight", type=float, default=5.0)
    apply_parser_defaults(parser, config_defaults or {})
    return parser


def main() -> None:
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", default=None)
    config_args, _ = config_probe.parse_known_args()
    config_defaults = load_yaml_defaults(config_args.config)
    parser = build_parser(config_defaults)
    args = parser.parse_args()
    if args.dataset_repo_id is None:
        parser.error("--dataset-repo-id is required unless it is set in --config")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    checkpoint = load_checkpoint(args.resume, map_location=device)
    config = config_from_args(args, checkpoint)
    config.device = str(device)

    dataset = load_lerobot_dataset(args)
    stats = dataset_stats(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=collate_batch,
    )
    val_dataloader = None
    if args.val_split_json is not None:
        val_dataset = load_lerobot_dataset(args, split_json=args.val_split_json, split_name=args.val_split_name)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=collate_batch,
        )

    policy = MiniVLAPolicy(config).to(device)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=args.lr, weight_decay=args.weight_decay)
    start_step = 0
    start_epoch = 0

    if checkpoint is not None:
        policy.load_compatible_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError as exc:
                print(f"warning: skipped incompatible optimizer state: {exc}", flush=True)
        start_step = int(checkpoint.get("step", 0))
        start_epoch = int(checkpoint.get("epoch", 0))

    processor = MiniVLAProcessor(config)
    normalizer = BatchNormalizer(
        stats,
        device=device,
        normalize_state=args.normalize_state,
        normalize_action=args.normalize_action,
    )

    output_dir = Path(args.output_dir)
    assets = {
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": args.dataset_root,
        "image_keys": list(config.image_keys),
        "optional_metadata_keys": [
            EPISODE_SUCCESS,
            EPISODE_QUALITY,
            SUBTASK_LABEL,
            SUBGOAL_IMAGE,
            FUTURE_IMAGE,
        ],
        "normalizer": {
            "normalize_state": args.normalize_state,
            "normalize_action": args.normalize_action,
            "stats_key": "norm_stats",
        },
        "normalize_state": args.normalize_state,
        "normalize_action": args.normalize_action,
        "processor": processor.metadata(),
    }
    max_steps = args.max_steps if args.max_steps is not None else math.inf
    step = start_step
    best_metric_value = math.inf
    failure_episode_weights = load_failure_episode_weights(args.failure_manifest, args.failure_upweight)
    jsonl_log = Path(args.train_log_jsonl) if args.train_log_jsonl is not None else output_dir / "train_log.jsonl"
    csv_log = Path(args.train_log_csv) if args.train_log_csv is not None else output_dir / "train_log.csv"

    policy.train()
    for epoch in range(start_epoch, args.epochs):
        for raw_batch in dataloader:
            batch = prepare_batch(raw_batch, config, processor, normalizer, device=device)
            sample_weights, weight_logs = build_sample_weights(raw_batch, batch, args, failure_episode_weights, device)
            use_custom_reduction = sample_weights is not None or args.loss_clip > 0
            if use_custom_reduction:
                loss_tensor, info = policy(batch, reduction="none")
                loss, loss_logs = reduce_sft_loss(
                    loss_tensor,
                    sample_weights,
                    args.loss_clip,
                    action_pad_mask=batch.get(ACTION_IS_PAD, batch.get("action_pad_mask", batch.get("actions_id_pad"))),
                )
                if "fm_action_smoothness_loss" in info:
                    loss = loss + config.fm_action_smoothness_loss_weight * info["fm_action_smoothness_loss"]
                if config.future_latent_loss_weight > 0 and FUTURE_IMAGE in batch:
                    future_tokens = policy.encode_observation_tokens(batch)
                    future_loss = policy.future_latent_loss(batch, future_tokens)
                    if future_loss is not None:
                        loss = loss + config.future_latent_loss_weight * future_loss
                        info["future_latent_loss"] = future_loss.detach()
                info = {
                    **info,
                    **loss_logs,
                    **weight_logs,
                    "loss": loss.detach(),
                }
            else:
                loss, info = policy(batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip_norm)
            optimizer.step()

            step += 1
            if step % args.log_every == 0:
                train_record = {
                    "type": "train",
                    "step": step,
                    "epoch": epoch,
                    "loss": float(info["loss"]),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                for key in (
                    "sample_loss_mean",
                    "sample_loss_max",
                    "loss_clip_fraction",
                    "sample_weight_mean",
                    "sample_weight_max",
                    "fm_action_smoothness_loss",
                    "future_latent_loss",
                    "fm_time_mean",
                    "fm_noise_std",
                ):
                    if key in info:
                        value = info[key]
                        train_record[key] = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
                write_jsonl(jsonl_log, train_record)
                write_csv(csv_log, train_record)
                print(f"step={step} epoch={epoch} loss={float(info['loss']):.6f}", flush=True)
            if val_dataloader is not None and args.eval_every > 0 and step % args.eval_every == 0:
                metrics = evaluate_policy(policy, val_dataloader, processor, normalizer, config, args, device)
                metric_value = metrics.get(args.best_metric)
                if metric_value is None:
                    raise KeyError(f"--best-metric {args.best_metric!r} is not in validation metrics: {sorted(metrics)}")
                eval_record = {
                    "type": "eval",
                    "step": step,
                    "epoch": epoch,
                    **{f"val_{key}": value for key, value in metrics.items()},
                }
                write_jsonl(jsonl_log, eval_record)
                write_csv(csv_log, eval_record)
                print(
                    f"eval step={step} metric={args.best_metric} value={metric_value:.6f} "
                    f"mse={metrics['mean_sampled_action_mse']:.6f}",
                    flush=True,
                )
                if metric_value < best_metric_value:
                    best_metric_value = metric_value
                    save_checkpoint(output_dir / "best.pt", policy, optimizer, step, epoch, config, stats, assets)
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(output_dir / f"step_{step:08d}.pt", policy, optimizer, step, epoch, config, stats, assets)
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    save_checkpoint(output_dir / "last.pt", policy, optimizer, step, epoch, config, stats, assets)
    if val_dataloader is not None and args.eval_every <= 0:
        metrics = evaluate_policy(policy, val_dataloader, processor, normalizer, config, args, device)
        save_checkpoint(output_dir / "best.pt", policy, optimizer, step, epoch, config, stats, assets)
        write_jsonl(
            jsonl_log,
            {"type": "eval", "step": step, "epoch": epoch, **{f"val_{key}": value for key, value in metrics.items()}},
        )
    print(f"saved checkpoint: {output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
