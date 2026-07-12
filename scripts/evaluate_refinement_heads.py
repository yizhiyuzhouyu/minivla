from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from minivla import MiniVLAPolicyRunner, PostSFTRefinementStack
from minivla.constants import ACTION
from minivla.postprocess import action_jerk, action_smoothness
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


def parse_refinement_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for item in specs:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(path).parent.name if Path(path).name in {"last.pt", "refinement.pt"} else Path(path).stem
        parsed.append((name, Path(path)))
    return parsed


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target[..., : pred.shape[-1]]) ** 2)


def horizon_labels(pred: torch.Tensor, target: torch.Tensor, choices: tuple[int, ...], threshold: float) -> torch.Tensor:
    per_step_error = (pred - target[..., : pred.shape[-1]]).square().mean(dim=-1)
    labels = []
    for sample_index in range(pred.shape[0]):
        chosen = 0
        for choice_index, choice in enumerate(choices):
            steps = min(choice, pred.shape[1])
            prefix_error = per_step_error[sample_index, :steps].mean()
            if float(prefix_error.detach().cpu()) <= threshold:
                chosen = choice_index
        labels.append(chosen)
    return torch.tensor(labels, dtype=torch.long, device=pred.device)


def failure_labels(
    pred: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    sample_mse = (pred - target[..., : pred.shape[-1]]).square().flatten(start_dim=1).mean(dim=1)
    sample_smoothness = (
        (pred[:, 1:] - pred[:, :-1]).abs().flatten(start_dim=1).mean(dim=1)
        if pred.shape[1] >= 2
        else torch.zeros(pred.shape[0], device=pred.device)
    )
    sample_jerk = (
        (pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]).abs().flatten(start_dim=1).mean(dim=1)
        if pred.shape[1] >= 3
        else torch.zeros(pred.shape[0], device=pred.device)
    )
    return (
        (sample_mse > args.failure_mse_threshold)
        | (sample_smoothness > args.failure_smoothness_threshold)
        | (sample_jerk > args.failure_jerk_threshold)
    ).float()


def binary_auroc(scores: list[float], labels: list[float]) -> float:
    positives = sum(1 for label in labels if label > 0.5)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label > 0.5:
            rank_sum += rank
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@torch.no_grad()
def evaluate_base(runner: MiniVLAPolicyRunner, dataloader: DataLoader, args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, list[float]] = {
        "mse": [],
        "smoothness": [],
        "jerk": [],
        "latency_ms": [],
    }
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
        obs_batch = {key: value for key, value in batch.items() if key != ACTION}
        if runner.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        pred = runner.policy.predict_action_chunk(obs_batch, num_steps=args.num_steps)
        if runner.device.type == "cuda":
            torch.cuda.synchronize()
        values["latency_ms"].append((time.perf_counter() - start) * 1000.0)
        target = batch[ACTION][..., : pred.shape[-1]]
        steps = min(pred.shape[1], target.shape[1])
        pred = pred[:, :steps].float()
        target = target[:, :steps].float()
        pred_original = runner.normalizer.unnormalize_actions(pred)
        target_original = runner.normalizer.unnormalize_actions(target)
        values["mse"].append(float(mse(pred_original, target_original).detach().cpu()))
        values["smoothness"].append(float(action_smoothness(pred_original).detach().cpu()))
        values["jerk"].append(float(action_jerk(pred_original).detach().cpu()))
    return {
        "name": "base",
        "action_metric_space": "original_action_space_after_unnormalize",
        "mean_sampled_action_mse": mean(values["mse"]),
        "mean_action_smoothness": mean(values["smoothness"]),
        "mean_action_jerk": mean(values["jerk"]),
        "latency_ms": mean(values["latency_ms"]),
    }


@torch.no_grad()
def evaluate_refinement(
    name: str,
    checkpoint: Path,
    runner: MiniVLAPolicyRunner,
    dataloader: DataLoader,
    args: argparse.Namespace,
) -> dict[str, Any]:
    stack = PostSFTRefinementStack.from_checkpoint(checkpoint, runner.policy.config, map_location=runner.device).to(runner.device)
    stack.eval()
    values: dict[str, list[float]] = {
        "base_mse": [],
        "refined_mse": [],
        "verifier_selected_mse": [],
        "base_smoothness": [],
        "refined_smoothness": [],
        "base_jerk": [],
        "refined_jerk": [],
        "failure_probability": [],
        "safety_probability": [],
        "horizon": [],
        "horizon_accuracy": [],
        "failure_scores": [],
        "failure_labels": [],
        "base_latency_ms": [],
        "refinement_latency_ms": [],
    }

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
        obs_batch = {key: value for key, value in batch.items() if key != ACTION}
        obs_tokens = runner.policy.encode_observation_tokens(obs_batch).float()
        if runner.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        pred = runner.policy.predict_action_chunk(obs_batch, num_steps=args.num_steps).float()
        if runner.device.type == "cuda":
            torch.cuda.synchronize()
        values["base_latency_ms"].append((time.perf_counter() - start) * 1000.0)

        target = batch[ACTION][..., : pred.shape[-1]].float()
        steps = min(pred.shape[1], target.shape[1])
        pred = pred[:, :steps]
        target = target[:, :steps]
        pred_original = runner.normalizer.unnormalize_actions(pred)
        target_original = runner.normalizer.unnormalize_actions(target)
        values["base_mse"].append(float(mse(pred_original, target_original).detach().cpu()))
        values["base_smoothness"].append(float(action_smoothness(pred_original).detach().cpu()))
        values["base_jerk"].append(float(action_jerk(pred_original).detach().cpu()))

        if runner.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = stack(obs_tokens, pred)
        if runner.device.type == "cuda":
            torch.cuda.synchronize()
        values["refinement_latency_ms"].append((time.perf_counter() - start) * 1000.0)

        refined = out["refined_actions"].float()
        refined_steps = min(refined.shape[1], target.shape[1])
        refined = refined[:, :refined_steps]
        target_refined = target[:, :refined_steps]
        refined_original = runner.normalizer.unnormalize_actions(refined)
        target_refined_original = runner.normalizer.unnormalize_actions(target_refined)
        values["refined_mse"].append(float(mse(refined_original, target_refined_original).detach().cpu()))
        values["refined_smoothness"].append(float(action_smoothness(refined_original).detach().cpu()))
        values["refined_jerk"].append(float(action_jerk(refined_original).detach().cpu()))

        if "probe" in out:
            values["failure_probability"].append(float(out["probe"]["failure_probability"].mean().detach().cpu()))
            labels = failure_labels(pred_original, target_original, args)
            values["failure_scores"].extend(out["probe"]["failure_probability"].detach().cpu().flatten().tolist())
            values["failure_labels"].extend(labels.detach().cpu().flatten().tolist())
        if "verifier" in out:
            values["safety_probability"].append(float(out["verifier"]["safety_probability"].mean().detach().cpu()))
        if "horizon" in out:
            horizon = out["horizon"]["horizon"]
            values["horizon"].append(float(horizon.float().mean().detach().cpu()))
            labels = horizon_labels(
                pred_original,
                target_original,
                tuple(stack.refinement_config.horizon_choices),
                args.horizon_mse_threshold,
            )
            horizon_choices = torch.tensor(stack.refinement_config.horizon_choices, dtype=torch.long, device=pred.device)
            label_horizon = horizon_choices[labels]
            values["horizon_accuracy"].append(float((horizon == label_horizon).float().mean().detach().cpu()))

        if args.num_resample_candidates > 1 and "verifier" in out:
            best_score = out["verifier"]["safety_probability"] + 0.1 * out["verifier"]["advantage"]
            best_actions = pred
            for _ in range(args.num_resample_candidates - 1):
                candidate = runner.policy.predict_action_chunk(obs_batch, num_steps=args.num_steps).float()[:, :steps]
                candidate_out = stack(obs_tokens, candidate)
                candidate_score = candidate_out["verifier"]["safety_probability"] + 0.1 * candidate_out["verifier"]["advantage"]
                mask = candidate_score > best_score
                best_score = torch.where(mask, candidate_score, best_score)
                best_actions = torch.where(mask[:, None, None], candidate, best_actions)
            best_actions_original = runner.normalizer.unnormalize_actions(best_actions)
            values["verifier_selected_mse"].append(float(mse(best_actions_original, target_original).detach().cpu()))

    return {
        "name": name,
        "checkpoint": str(checkpoint),
        "action_metric_space": "original_action_space_after_unnormalize",
        "base_mse": mean(values["base_mse"]),
        "refined_mse": mean(values["refined_mse"]),
        "verifier_selected_mse": mean(values["verifier_selected_mse"]),
        "base_smoothness": mean(values["base_smoothness"]),
        "refined_smoothness": mean(values["refined_smoothness"]),
        "base_jerk": mean(values["base_jerk"]),
        "refined_jerk": mean(values["refined_jerk"]),
        "mean_failure_probability": mean(values["failure_probability"]),
        "mean_safety_probability": mean(values["safety_probability"]),
        "mean_horizon": mean(values["horizon"]),
        "horizon_accuracy": mean(values["horizon_accuracy"]),
        "probe_auroc": binary_auroc(values["failure_scores"], values["failure_labels"]),
        "base_latency_ms": mean(values["base_latency_ms"]),
        "refinement_latency_ms": mean(values["refinement_latency_ms"]),
        "latency_overhead_ms": mean(values["refinement_latency_ms"]),
        "residual_mse_delta": mean(values["refined_mse"]) - mean(values["base_mse"]),
    }


def markdown_table(report: dict[str, Any]) -> str:
    lines = [
        "# Refinement Ablation Report",
        "",
        "| name | base_mse | refined_mse | verifier_selected_mse | mse_delta | probe_auroc | base_jerk | refined_jerk | mean_horizon | horizon_acc | overhead_ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["refinements"]:
        lines.append(
            "| {name} | {base_mse:.6f} | {refined_mse:.6f} | {verifier_selected_mse:.6f} | "
            "{residual_mse_delta:.6f} | {probe_auroc:.3f} | {base_jerk:.6f} | {refined_jerk:.6f} | "
            "{mean_horizon:.2f} | {horizon_accuracy:.3f} | {latency_overhead_ms:.3f} |".format(**item)
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate post-SFT refinement head ablations.")
    parser.add_argument("--checkpoint", required=True, help="Base MiniVLA checkpoint.")
    parser.add_argument("--refinement", action="append", default=[], help="name=checkpoint or checkpoint path.")
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
    parser.add_argument("--delta-timestamps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--num-resample-candidates", type=int, default=1)
    parser.add_argument("--horizon-mse-threshold", type=float, default=0.5)
    parser.add_argument("--failure-mse-threshold", type=float, default=1.0)
    parser.add_argument("--failure-smoothness-threshold", type=float, default=0.5)
    parser.add_argument("--failure-jerk-threshold", type=float, default=1.0)
    parser.add_argument("--output-dir", default="outputs/refinement_ablation")
    args = parser.parse_args()

    runner = MiniVLAPolicyRunner.from_checkpoint(args.checkpoint, device=args.device)
    dataset = load_lerobot_dataset(args, runner)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=runner.device.type == "cuda",
        drop_last=False,
        collate_fn=collate_batch,
    )

    base = evaluate_base(runner, dataloader, args)
    refinements = []
    for name, path in parse_refinement_specs(args.refinement):
        metrics = evaluate_refinement(name, path, runner, dataloader, args)
        refinements.append(metrics)
        print(
            f"name={name} base_mse={metrics['base_mse']:.6f} "
            f"refined_mse={metrics['refined_mse']:.6f} overhead_ms={metrics['latency_overhead_ms']:.3f}"
        )

    report = {
        "base_checkpoint": args.checkpoint,
        "base": base,
        "refinements": refinements,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with (output_dir / "table.md").open("w", encoding="utf-8") as handle:
        handle.write(markdown_table(report))
    print(f"saved_report={output_dir / 'report.json'}")
    print(f"saved_table={output_dir / 'table.md'}")


if __name__ == "__main__":
    main()
