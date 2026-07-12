from __future__ import annotations

import argparse
import itertools
import json
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
from minivla.postprocess import ActionPostProcessor, PostProcessConfig, action_jerk, action_smoothness
from minivla.splits import load_episode_split
from minivla.transforms import prepare_batch
from train import build_delta_timestamps, collate_batch


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


def parse_floats(values: list[str]) -> list[float]:
    return [float(value) for value in values]


def parse_ints(values: list[str]) -> list[int]:
    return [int(value) for value in values]


def candidate_grid(args: argparse.Namespace):
    return itertools.product(
        parse_ints(args.num_inference_steps),
        parse_ints(args.n_action_steps),
        parse_floats(args.ema_alpha),
        parse_floats(args.max_delta),
        parse_floats(args.action_min),
        parse_floats(args.action_max),
        parse_floats(args.temporal_ensemble_decay),
    )


@torch.no_grad()
def evaluate_candidate(
    runner: MiniVLAPolicyRunner,
    dataloader: DataLoader,
    args: argparse.Namespace,
    num_steps: int,
    n_action_steps: int,
    ema_alpha: float,
    max_delta: float,
    action_min: float,
    action_max: float,
    temporal_ensemble_decay: float,
) -> dict[str, Any]:
    if action_min > action_max:
        raise ValueError("action_min cannot exceed action_max")

    mse_values: list[float] = []
    smoothness_values: list[float] = []
    jerk_values: list[float] = []
    saturation_values: list[float] = []
    jump_values: list[float] = []
    nonfinite_values: list[float] = []
    latencies_ms: list[float] = []

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
        if runner.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        pred = runner.policy.predict_action_chunk({key: value for key, value in batch.items() if key != ACTION}, num_steps=num_steps)
        if runner.device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        target = batch[ACTION][..., : pred.shape[-1]]
        steps = min(n_action_steps, pred.shape[1], target.shape[1])
        pred = pred[:, :steps]
        target = target[:, :steps]
        pred = runner.normalizer.unnormalize_actions(pred)
        target = runner.normalizer.unnormalize_actions(target)

        processed_samples = []
        infos = []
        for sample_index in range(pred.shape[0]):
            processor = ActionPostProcessor(
                PostProcessConfig(
                    action_dim=runner.policy.config.action_dim,
                    action_min=action_min,
                    action_max=action_max,
                    max_delta=max_delta,
                    ema_alpha=ema_alpha,
                )
            )
            processed_steps = []
            for step_index in range(steps):
                processed, info = processor(pred[sample_index, step_index])
                processed_steps.append(processed)
                infos.append(info)
            processed_samples.append(torch.stack(processed_steps, dim=0))
        processed_pred = torch.stack(processed_samples, dim=0)
        target = target[..., : processed_pred.shape[-1]]

        mse_values.append(float(torch.mean((processed_pred - target) ** 2).detach().cpu()))
        smoothness_values.append(float(action_smoothness(processed_pred).detach().cpu()))
        jerk_values.append(float(action_jerk(processed_pred).detach().cpu()))
        if infos:
            saturation_values.append(sum(info.saturation_ratio for info in infos) / len(infos))
            jump_values.append(sum(info.command_jump_ratio for info in infos) / len(infos))
            nonfinite_values.append(sum(info.nonfinite_count for info in infos) / len(infos))

    if not mse_values:
        raise RuntimeError("No batches evaluated")

    metrics = {
        "num_inference_steps": num_steps,
        "n_action_steps": n_action_steps,
        "action_metric_space": "original_action_space_after_unnormalize",
        "use_temporal_ensemble": args.use_temporal_ensemble,
        "temporal_ensemble_decay": temporal_ensemble_decay,
        "ema_alpha": ema_alpha,
        "max_delta": max_delta,
        "action_min": action_min,
        "action_max": action_max,
        "mean_sampled_action_mse": sum(mse_values) / len(mse_values),
        "mean_action_smoothness": sum(smoothness_values) / len(smoothness_values),
        "mean_action_jerk": sum(jerk_values) / len(jerk_values),
        "action_saturation_ratio": sum(saturation_values) / len(saturation_values) if saturation_values else 0.0,
        "command_jump_ratio": sum(jump_values) / len(jump_values) if jump_values else 0.0,
        "nonfinite_per_action": sum(nonfinite_values) / len(nonfinite_values) if nonfinite_values else 0.0,
        "latency_ms": sum(latencies_ms) / len(latencies_ms),
    }
    metrics["selection_score"] = (
        metrics["mean_sampled_action_mse"]
        + args.smoothness_weight * metrics["mean_action_smoothness"]
        + args.jerk_weight * metrics["mean_action_jerk"]
        + args.saturation_weight * metrics["action_saturation_ratio"]
        + args.jump_weight * metrics["command_jump_ratio"]
        + args.nonfinite_weight * metrics["nonfinite_per_action"]
        + args.latency_weight * metrics["latency_ms"]
    )
    return metrics


def pareto_frontier(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    objectives = (
        "mean_sampled_action_mse",
        "mean_action_smoothness",
        "mean_action_jerk",
        "action_saturation_ratio",
        "command_jump_ratio",
        "latency_ms",
    )
    frontier = []
    for candidate in results:
        dominated = False
        for other in results:
            if other is candidate:
                continue
            no_worse = all(other[key] <= candidate[key] for key in objectives)
            strictly_better = any(other[key] < candidate[key] for key in objectives)
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda item: item["selection_score"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate post-SFT MiniVLA inference and action postprocess parameters.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--split-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/post_sft")
    parser.add_argument("--delta-timestamps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fps", type=float, default=30.0)

    parser.add_argument("--num-inference-steps", nargs="+", default=["2", "4", "8", "10", "16"])
    parser.add_argument("--n-action-steps", nargs="+", default=["25"])
    parser.add_argument("--temporal-ensemble-decay", nargs="+", default=["0.5"])
    parser.add_argument("--use-temporal-ensemble", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-alpha", nargs="+", default=["0.2", "0.35", "0.5"])
    parser.add_argument("--max-delta", nargs="+", default=["0.1", "0.2", "0.35"])
    parser.add_argument("--action-min", nargs="+", default=["-1.0"])
    parser.add_argument("--action-max", nargs="+", default=["1.0"])

    parser.add_argument("--smoothness-weight", type=float, default=0.1)
    parser.add_argument("--jerk-weight", type=float, default=0.1)
    parser.add_argument("--saturation-weight", type=float, default=1.0)
    parser.add_argument("--jump-weight", type=float, default=1.0)
    parser.add_argument("--nonfinite-weight", type=float, default=10.0)
    parser.add_argument("--latency-weight", type=float, default=0.001)
    args = parser.parse_args()

    runner = MiniVLAPolicyRunner.from_checkpoint(args.checkpoint, device=args.device)
    dataset = load_lerobot_dataset(args, runner)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_batch,
    )

    results = []
    for values in candidate_grid(args):
        num_steps, n_action_steps, ema_alpha, max_delta, action_min, action_max, temporal_ensemble_decay = values
        if action_min > action_max:
            continue
        metrics = evaluate_candidate(
            runner,
            dataloader,
            args,
            num_steps=num_steps,
            n_action_steps=n_action_steps,
            ema_alpha=ema_alpha,
            max_delta=max_delta,
            action_min=action_min,
            action_max=action_max,
            temporal_ensemble_decay=temporal_ensemble_decay,
        )
        results.append(metrics)
        print(
            f"steps={num_steps} n_action_steps={n_action_steps} "
            f"ema={ema_alpha} max_delta={max_delta} "
            f"score={metrics['selection_score']:.6f} "
            f"mse={metrics['mean_sampled_action_mse']:.6f} "
            f"latency_ms={metrics['latency_ms']:.3f}"
        )

    if not results:
        raise RuntimeError("No calibration candidates evaluated")

    results.sort(key=lambda item: item["selection_score"])
    frontier = pareto_frontier(results)
    recommended = results[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": args.checkpoint,
        "selection_rule": "lower selection_score is better; pareto_frontier keeps non-dominated candidates",
        "recommended": recommended,
        "pareto_frontier": frontier,
        "results": results,
        "notes": [
            "Offline calibration evaluates chunks from validation observations.",
            "Action metrics and postprocess constraints are evaluated in original action space after unnormalizing checkpoint outputs and targets.",
            "Temporal ensemble decay is recorded for deployment configs; closed-loop robot rollout is still needed to validate temporal ensembling.",
        ],
    }
    with (output_dir / "calibration_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with (output_dir / "recommended_post_sft_config.yaml").open("w", encoding="utf-8") as handle:
        for key in (
            "num_inference_steps",
            "n_action_steps",
            "use_temporal_ensemble",
            "temporal_ensemble_decay",
            "ema_alpha",
            "max_delta",
            "action_min",
            "action_max",
        ):
            handle.write(f"{key}: {recommended[key]}\n")

    print(f"recommended={recommended}")
    print(f"saved_report={output_dir / 'calibration_report.json'}")
    print(f"saved_config={output_dir / 'recommended_post_sft_config.yaml'}")


if __name__ == "__main__":
    main()
