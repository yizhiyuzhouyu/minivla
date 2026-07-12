from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from minivla import MiniVLAPolicyRunner
from minivla.constants import ACTION
from minivla.postprocess import action_jerk, action_saturation_ratio, action_smoothness, command_jump_ratio
from minivla.splits import load_episode_split
from minivla.transforms import prepare_batch
from train import build_delta_timestamps, collate_batch, per_sample_sft_loss


def load_lerobot_dataset(args: argparse.Namespace, runner: MiniVLAPolicyRunner):
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise ImportError("Could not import LeRobotDataset. Install LeRobot first.") from exc

    episodes = args.episodes
    split_episodes = load_episode_split(args.split_json, args.split_name)
    if split_episodes is not None:
        episodes = split_episodes
    kwargs: dict[str, Any] = {}
    if args.dataset_root is not None:
        kwargs["root"] = args.dataset_root
    if episodes is not None:
        kwargs["episodes"] = episodes
    if args.delta_timestamps:
        kwargs["delta_timestamps"] = build_delta_timestamps(
            runner.policy.config.chunk_size,
            runner.policy.config.n_obs_steps,
            args.fps,
            runner.policy.config.image_keys,
        )
    return LeRobotDataset(args.dataset_repo_id, **kwargs)


def checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(item) for item in args.checkpoints]
    for pattern in args.checkpoint_glob:
        paths.extend(Path(item) for item in sorted(glob.glob(pattern)))
    resolved = []
    for path in paths:
        if path.is_dir():
            path = path / "policy.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        resolved.append(path)
    if not resolved:
        raise ValueError("Provide at least one checkpoint via --checkpoints or --checkpoint-glob")
    return resolved


def _sample_ids(raw_batch: dict[str, Any], index: int) -> dict[str, Any]:
    out: dict[str, Any] = {"sample_index": index}
    for key in ("episode_index", "episode.id", "episode", "frame_index", "timestamp"):
        value = raw_batch.get(key)
        if value is None:
            continue
        if torch.is_tensor(value):
            if value.ndim == 0:
                out[key] = value.item()
            elif index < value.shape[0]:
                item = value[index]
                out[key] = item.item() if item.numel() == 1 else item.detach().cpu().tolist()
        elif isinstance(value, list) and index < len(value):
            out[key] = value[index]
        else:
            out[key] = value
    return out


def _classify_failure(
    sample_loss: float,
    sample_mse: float,
    sample_smoothness: float,
    sample_jerk: float,
    sample_saturation: float,
    sample_jump: float,
    args: argparse.Namespace,
) -> list[str]:
    tags = []
    if sample_loss >= args.loss_threshold:
        tags.append("high_loss")
    if sample_mse >= args.action_mse_threshold:
        tags.append("high_action_mse")
    if sample_smoothness >= args.smoothness_threshold:
        tags.append("action_jump")
    if sample_jerk >= args.jerk_threshold:
        tags.append("high_jerk")
    if sample_saturation >= args.saturation_threshold:
        tags.append("action_saturation")
    if sample_jump >= args.command_jump_threshold:
        tags.append("command_jump")
    return tags


@torch.no_grad()
def evaluate_checkpoint(path: Path, dataloader: DataLoader, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    runner = MiniVLAPolicyRunner.from_checkpoint(path, device=args.device)
    losses: list[float] = []
    action_mses: list[float] = []
    action_mses_normalized: list[float] = []
    smoothness_values: list[float] = []
    jerk_values: list[float] = []
    saturation_values: list[float] = []
    jump_values: list[float] = []
    latencies_ms: list[float] = []
    per_action_dim_sum: torch.Tensor | None = None
    per_action_dim_count = 0
    failure_cases: list[dict[str, Any]] = []
    per_episode: dict[str, dict[str, float | int]] = {}

    for batch_index, raw_batch in enumerate(dataloader):
        if batch_index >= args.max_batches:
            break
        batch = prepare_batch(
            raw_batch,
            runner.policy.config,
            runner.processor,
            runner.normalizer,
            device=runner.device,
            require_action=True,
        )

        loss_none, _ = runner.policy(batch, reduction="none")
        sample_loss_tensor, _ = per_sample_sft_loss(
            loss_none,
            action_pad_mask=batch.get("action_is_pad", batch.get("action_pad_mask", batch.get("actions_id_pad"))),
        )
        loss = sample_loss_tensor.mean()
        losses.append(float(loss.detach().cpu()))

        if runner.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        pred = runner.policy.predict_action_chunk({key: value for key, value in batch.items() if key != ACTION}, num_steps=args.num_steps)
        if runner.device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        target = batch[ACTION][..., : pred.shape[-1]]
        steps = min(pred.shape[1], target.shape[1])
        pred = pred[:, :steps]
        target = target[:, :steps]
        squared_error_normalized = (pred - target) ** 2
        sample_mse_normalized_tensor = squared_error_normalized.flatten(start_dim=1).mean(dim=1)
        action_mses_normalized.append(float(sample_mse_normalized_tensor.mean().detach().cpu()))
        pred = runner.normalizer.unnormalize_actions(pred)
        target = runner.normalizer.unnormalize_actions(target)
        squared_error = (pred - target) ** 2
        sample_mse_tensor = squared_error.flatten(start_dim=1).mean(dim=1)
        action_mses.append(float(sample_mse_tensor.mean().detach().cpu()))

        dim_sum = squared_error.sum(dim=(0, 1)).detach().float().cpu()
        per_action_dim_sum = dim_sum if per_action_dim_sum is None else per_action_dim_sum + dim_sum
        per_action_dim_count += squared_error.shape[0] * squared_error.shape[1]

        sample_smoothness_tensor = (pred[:, 1:] - pred[:, :-1]).abs().flatten(start_dim=1).mean(dim=1) if steps >= 2 else torch.zeros(pred.shape[0], device=pred.device)
        sample_jerk_tensor = (
            (pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]).abs().flatten(start_dim=1).mean(dim=1)
            if steps >= 3
            else torch.zeros(pred.shape[0], device=pred.device)
        )
        sample_saturation_tensor = ((pred <= args.action_min + args.saturation_eps) | (pred >= args.action_max - args.saturation_eps)).float().flatten(start_dim=1).mean(dim=1)
        sample_jump_tensor = (
            ((pred[:, 1:] - pred[:, :-1]).abs() > args.max_delta).float().flatten(start_dim=1).mean(dim=1)
            if steps >= 2 and args.max_delta > 0
            else torch.zeros(pred.shape[0], device=pred.device)
        )

        smoothness_values.append(float(action_smoothness(pred).detach().cpu()))
        jerk_values.append(float(action_jerk(pred).detach().cpu()))
        saturation_values.append(float(action_saturation_ratio(pred, args.action_min, args.action_max, args.saturation_eps).detach().cpu()))
        jump_values.append(float(command_jump_ratio(pred, args.max_delta).detach().cpu()) if args.max_delta > 0 else 0.0)

        for sample_index in range(pred.shape[0]):
            sample_loss = float(sample_loss_tensor[sample_index].detach().cpu())
            sample_mse = float(sample_mse_tensor[sample_index].detach().cpu())
            sample_smoothness = float(sample_smoothness_tensor[sample_index].detach().cpu())
            sample_jerk = float(sample_jerk_tensor[sample_index].detach().cpu())
            sample_saturation = float(sample_saturation_tensor[sample_index].detach().cpu())
            sample_jump = float(sample_jump_tensor[sample_index].detach().cpu())
            ids = _sample_ids(raw_batch, sample_index)
            episode_key = str(ids.get("episode_index", ids.get("episode.id", ids.get("episode", "unknown"))))
            episode_stats = per_episode.setdefault(episode_key, {"count": 0, "loss_sum": 0.0, "mse_sum": 0.0})
            episode_stats["count"] = int(episode_stats["count"]) + 1
            episode_stats["loss_sum"] = float(episode_stats["loss_sum"]) + sample_loss
            episode_stats["mse_sum"] = float(episode_stats["mse_sum"]) + sample_mse

            tags = _classify_failure(
                sample_loss,
                sample_mse,
                sample_smoothness,
                sample_jerk,
                sample_saturation,
                sample_jump,
                args,
            )
            score = sample_loss + sample_mse + 0.1 * sample_smoothness + 0.1 * sample_jerk + sample_saturation + sample_jump
            if tags or len(failure_cases) < args.failure_top_k:
                failure_cases.append(
                    {
                        "checkpoint": str(path),
                        "batch_index": batch_index,
                        **ids,
                        "score": score,
                        "tags": tags or ["top_score"],
                        "loss": sample_loss,
                        "sampled_action_mse": sample_mse,
                        "sampled_action_mse_original": sample_mse,
                        "sampled_action_mse_normalized": float(sample_mse_normalized_tensor[sample_index].detach().cpu()),
                        "action_smoothness": sample_smoothness,
                        "action_jerk": sample_jerk,
                        "action_saturation_ratio": sample_saturation,
                        "command_jump_ratio": sample_jump,
                    }
                )

    if not losses:
        raise RuntimeError(f"No batches evaluated for {path}")

    per_episode_report = []
    for episode, values in per_episode.items():
        count = max(1, int(values["count"]))
        per_episode_report.append(
            {
                "episode": episode,
                "count": count,
                "mean_loss": float(values["loss_sum"]) / count,
                "mean_sampled_action_mse": float(values["mse_sum"]) / count,
            }
        )
    per_episode_report.sort(key=lambda item: item["mean_loss"], reverse=True)

    per_action_dim_mse = []
    if per_action_dim_sum is not None and per_action_dim_count > 0:
        per_action_dim_mse = (per_action_dim_sum / per_action_dim_count).tolist()

    metrics = {
        "checkpoint": str(path),
        "batches": len(losses),
        "action_head": runner.policy.config.action_head,
        "action_metric_space": "original_action_space_after_unnormalize",
        "mean_fm_loss": sum(losses) / len(losses),
        "mean_sampled_action_mse": sum(action_mses) / len(action_mses),
        "mean_sampled_action_mse_original": sum(action_mses) / len(action_mses),
        "mean_sampled_action_mse_normalized": sum(action_mses_normalized) / len(action_mses_normalized),
        "mean_action_smoothness": sum(smoothness_values) / len(smoothness_values),
        "mean_action_jerk": sum(jerk_values) / len(jerk_values),
        "action_saturation_ratio": sum(saturation_values) / len(saturation_values),
        "command_jump_ratio": sum(jump_values) / len(jump_values),
        "latency_ms": sum(latencies_ms) / len(latencies_ms),
        "per_action_dim_mse": per_action_dim_mse,
        "per_episode": per_episode_report[: args.per_episode_top_k],
    }
    metrics["selection_score"] = (
        metrics["mean_fm_loss"]
        + metrics["mean_sampled_action_mse"]
        + args.smoothness_weight * metrics["mean_action_smoothness"]
        + args.jerk_weight * metrics["mean_action_jerk"]
        + args.saturation_weight * metrics["action_saturation_ratio"]
        + args.jump_weight * metrics["command_jump_ratio"]
        + args.latency_weight * metrics["latency_ms"]
    )
    failure_cases.sort(key=lambda item: item["score"], reverse=True)
    return metrics, failure_cases[: args.failure_top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a post-SFT MiniVLA checkpoint with a validation scorecard.")
    parser.add_argument("--checkpoints", nargs="*", default=[], help="Checkpoint files or pretrained directories.")
    parser.add_argument("--checkpoint-glob", action="append", default=[], help="Glob pattern for checkpoint files.")
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--split-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/post_sft")
    parser.add_argument("--delta-timestamps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fps", type=float, default=30.0)

    parser.add_argument("--action-min", type=float, default=-1.0)
    parser.add_argument("--action-max", type=float, default=1.0)
    parser.add_argument("--max-delta", type=float, default=0.2)
    parser.add_argument("--saturation-eps", type=float, default=1e-6)
    parser.add_argument("--loss-threshold", type=float, default=1.0)
    parser.add_argument("--action-mse-threshold", type=float, default=1.0)
    parser.add_argument("--smoothness-threshold", type=float, default=0.5)
    parser.add_argument("--jerk-threshold", type=float, default=1.0)
    parser.add_argument("--saturation-threshold", type=float, default=0.05)
    parser.add_argument("--command-jump-threshold", type=float, default=0.05)
    parser.add_argument("--failure-top-k", type=int, default=50)
    parser.add_argument("--per-episode-top-k", type=int, default=20)

    parser.add_argument("--smoothness-weight", type=float, default=0.1)
    parser.add_argument("--jerk-weight", type=float, default=0.1)
    parser.add_argument("--saturation-weight", type=float, default=1.0)
    parser.add_argument("--jump-weight", type=float, default=1.0)
    parser.add_argument("--latency-weight", type=float, default=0.001)
    args = parser.parse_args()

    paths = checkpoint_paths(args)
    config_runner = MiniVLAPolicyRunner.from_checkpoint(paths[0], device=args.device)
    dataset = load_lerobot_dataset(args, config_runner)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_batch,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    all_failures = []
    for checkpoint in paths:
        metrics, failures = evaluate_checkpoint(checkpoint, dataloader, args)
        results.append(metrics)
        all_failures.extend(failures)
        print(
            f"checkpoint={checkpoint} "
            f"score={metrics['selection_score']:.6f} "
            f"loss={metrics['mean_fm_loss']:.6f} "
            f"mse={metrics['mean_sampled_action_mse']:.6f}"
        )

    results.sort(key=lambda item: item["selection_score"])
    best = results[0]
    best_path = Path(best["checkpoint"])
    shutil.copy2(best_path, output_dir / "best.pt")

    report = {
        "best_checkpoint": best["checkpoint"],
        "best_output": str(output_dir / "best.pt"),
        "selection_rule": "lower selection_score is better; action-space metrics are computed after unnormalizing checkpoint outputs and targets",
        "results": results,
    }
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    all_failures.sort(key=lambda item: item["score"], reverse=True)
    with (output_dir / "failure_cases.jsonl").open("w", encoding="utf-8") as handle:
        for item in all_failures[: args.failure_top_k]:
            handle.write(json.dumps(item) + "\n")

    print(f"best_checkpoint={best['checkpoint']}")
    print(f"saved_report={output_dir / 'report.json'}")
    print(f"saved_best={output_dir / 'best.pt'}")
    print(f"saved_failure_cases={output_dir / 'failure_cases.jsonl'}")


if __name__ == "__main__":
    main()
