from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from minivla import MiniVLAPolicyRunner, PostSFTRefinementStack, RefinementConfig
from minivla.constants import ACTION
from minivla.postprocess import action_jerk, action_smoothness, command_jump_ratio
from minivla.splits import load_episode_split
from minivla.transforms import prepare_batch
from train import build_delta_timestamps, collate_batch


FAILURE_LABELS = {"failure", "fail", "failed", "unsafe", "collision", "intervention", "abort", "stop", "0", "false"}
SUCCESS_LABELS = {"success", "succeeded", "safe", "ok", "pass", "1", "true"}


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


def refinement_config_from_args(args: argparse.Namespace) -> RefinementConfig:
    presets = {
        "probe": dict(
            enable_action_probe=True,
            enable_verifier=False,
            enable_adaptive_horizon=False,
            enable_residual_recovery=False,
        ),
        "probe_verifier": dict(
            enable_action_probe=True,
            enable_verifier=True,
            enable_adaptive_horizon=False,
            enable_residual_recovery=False,
        ),
        "probe_verifier_horizon": dict(
            enable_action_probe=True,
            enable_verifier=True,
            enable_adaptive_horizon=True,
            enable_residual_recovery=False,
        ),
        "full": dict(
            enable_action_probe=True,
            enable_verifier=True,
            enable_adaptive_horizon=True,
            enable_residual_recovery=True,
        ),
    }
    config = presets[args.preset]
    return RefinementConfig(
        **config,
        horizon_choices=tuple(args.horizon_choices),
        residual_scale=args.residual_scale,
        max_delta=args.max_delta,
        action_min=args.action_min,
        action_max=args.action_max,
    )


def failure_labels(
    pred: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    sample_mse = (pred - target).square().flatten(start_dim=1).mean(dim=1)
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
    sample_jump = (
        ((pred[:, 1:] - pred[:, :-1]).abs() > args.max_delta).float().flatten(start_dim=1).mean(dim=1)
        if pred.shape[1] >= 2
        else torch.zeros(pred.shape[0], device=pred.device)
    )
    labels = (
        (sample_mse > args.failure_mse_threshold)
        | (sample_smoothness > args.failure_smoothness_threshold)
        | (sample_jerk > args.failure_jerk_threshold)
        | (sample_jump > args.failure_jump_threshold)
    ).float()
    return labels, {
        "sample_mse": sample_mse.detach(),
        "sample_smoothness": sample_smoothness.detach(),
        "sample_jerk": sample_jerk.detach(),
        "sample_jump": sample_jump.detach(),
    }


def horizon_labels(pred: torch.Tensor, target: torch.Tensor, choices: tuple[int, ...], threshold: float) -> torch.Tensor:
    per_step_error = (pred - target).square().mean(dim=-1)
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


def _sample_value(raw_batch: dict[str, Any], key: str, index: int) -> Any:
    value = raw_batch.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        if index >= value.shape[0]:
            return None
        item = value[index]
        return item.item() if item.numel() == 1 else item.detach().cpu().tolist()
    if isinstance(value, list):
        return value[index] if index < len(value) else None
    return value


def _sample_key(raw_batch: dict[str, Any], index: int) -> tuple[str | None, int | None]:
    episode = None
    for key in ("episode_id", "episode_index", "episode.id", "episode"):
        value = _sample_value(raw_batch, key, index)
        if value is not None:
            episode = str(value)
            break
    frame = _sample_value(raw_batch, "frame_index", index)
    if frame is None:
        frame = _sample_value(raw_batch, "cycle", index)
    try:
        frame_int = int(frame) if frame is not None and not isinstance(frame, list) else None
    except (TypeError, ValueError):
        frame_int = None
    return episode, frame_int


def _rollout_failure_label(record: dict[str, Any], args: argparse.Namespace) -> float | None:
    label = record.get("human_label")
    if label is not None:
        label_text = str(label).strip().lower()
        if label_text in FAILURE_LABELS:
            return 1.0
        if label_text in SUCCESS_LABELS:
            return 0.0
    post = record.get("postprocess")
    if isinstance(post, dict):
        if int(post.get("nonfinite_count", 0)) > 0:
            return 1.0
        if bool(post.get("joint_limit_projection", False)):
            return 1.0
        if float(post.get("command_jump_ratio", 0.0)) > args.rollout_jump_threshold:
            return 1.0
        if float(post.get("saturation_ratio", 0.0)) > args.rollout_saturation_threshold:
            return 1.0
    return None


def load_rollout_labels(args: argparse.Namespace) -> dict[tuple[str, int], float]:
    if args.rollout_jsonl is None:
        return {}
    labels: dict[tuple[str, int], float] = {}
    with Path(args.rollout_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            episode = record.get("episode_id", record.get("episode_index", record.get("episode")))
            frame = record.get("frame_index", record.get("cycle"))
            label = _rollout_failure_label(record, args)
            if episode is None or frame is None or label is None:
                continue
            try:
                labels[(str(episode), int(frame))] = label
            except (TypeError, ValueError):
                continue
    return labels


def apply_rollout_label_overrides(
    raw_batch: dict[str, Any],
    labels: dict[tuple[str, int], float],
    failure_label: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    if not labels:
        return failure_label, 0.0
    values = failure_label.clone()
    matched = 0
    for index in range(values.shape[0]):
        episode, frame = _sample_key(raw_batch, index)
        if episode is None or frame is None:
            continue
        label = labels.get((episode, frame))
        if label is None:
            continue
        values[index] = float(label)
        matched += 1
    return values, matched / max(1, values.shape[0])


def save_checkpoint(
    path: Path,
    stack: PostSFTRefinementStack,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    refinement_config: RefinementConfig,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "refinement_model": stack.state_dict(),
            "optimizer": optimizer.state_dict(),
            "base_checkpoint": args.checkpoint,
            "refinement_config": asdict(refinement_config),
            "step": step,
            "assets": {
                "dataset_repo_id": args.dataset_repo_id,
                "preset": args.preset,
                "label_source": args.label_source,
            },
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train post-SFT MiniVLA refinement heads on frozen policy outputs.")
    parser.add_argument("--checkpoint", required=True, help="Base MiniVLA checkpoint.")
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--split-name", default=None)
    parser.add_argument("--output-dir", default="outputs/post_sft_refinement")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--delta-timestamps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--preset",
        choices=["probe", "probe_verifier", "probe_verifier_horizon", "full"],
        default="probe",
    )
    parser.add_argument("--horizon-choices", type=int, nargs="+", default=[1, 2, 4, 8, 16, 25])
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--max-delta", type=float, default=0.2)
    parser.add_argument("--action-min", type=float, default=-1.0)
    parser.add_argument("--action-max", type=float, default=1.0)
    parser.add_argument("--failure-mse-threshold", type=float, default=1.0)
    parser.add_argument("--failure-smoothness-threshold", type=float, default=0.5)
    parser.add_argument("--failure-jerk-threshold", type=float, default=1.0)
    parser.add_argument("--failure-jump-threshold", type=float, default=0.05)
    parser.add_argument("--rollout-jsonl", default=None, help="Optional rollout log with human_label/safety events.")
    parser.add_argument("--rollout-saturation-threshold", type=float, default=0.1)
    parser.add_argument("--rollout-jump-threshold", type=float, default=0.1)
    parser.add_argument("--horizon-mse-threshold", type=float, default=0.5)
    parser.add_argument("--verifier-loss-weight", type=float, default=1.0)
    parser.add_argument("--horizon-loss-weight", type=float, default=1.0)
    parser.add_argument("--residual-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--label-source",
        default="pseudo_labels_from_sft_prediction_error",
        choices=[
            "pseudo_labels_from_sft_prediction_error",
            "rollout_human_labels",
            "rollout_success_failure",
            "heuristic_safety_events",
        ],
    )
    args = parser.parse_args()

    runner = MiniVLAPolicyRunner.from_checkpoint(args.checkpoint, device=args.device)
    for param in runner.policy.parameters():
        param.requires_grad = False
    dataset = load_lerobot_dataset(args, runner)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=runner.device.type == "cuda",
        drop_last=True,
        collate_fn=collate_batch,
    )

    refinement_config = refinement_config_from_args(args)
    stack = PostSFTRefinementStack(runner.policy.config, refinement_config).to(runner.device)
    optimizer = torch.optim.AdamW(stack.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rollout_labels = load_rollout_labels(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "refinement_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(refinement_config), handle, indent=2)

    step = 0
    max_steps = args.max_steps if args.max_steps is not None else float("inf")
    runner.policy.eval()
    stack.train()

    for epoch in range(args.epochs):
        for raw_batch in dataloader:
            batch = prepare_batch(
                raw_batch,
                runner.policy.config,
                runner.processor,
                runner.normalizer,
                device=runner.device,
                require_action=True,
            )
            with torch.no_grad():
                obs_batch = {key: value for key, value in batch.items() if key != ACTION}
                obs_tokens = runner.policy.encode_observation_tokens(obs_batch)
                pred = runner.policy.predict_action_chunk(obs_batch)
                target = batch[ACTION][..., : pred.shape[-1]]
                steps = min(pred.shape[1], target.shape[1])
                obs_tokens = obs_tokens.float()
                pred = pred[:, :steps].float()
                target = target[:, :steps].float()
                pred_original = runner.normalizer.unnormalize_actions(pred)
                target_original = runner.normalizer.unnormalize_actions(target)
                failure_label, diagnostics = failure_labels(pred_original, target_original, args)
                if args.label_source != "pseudo_labels_from_sft_prediction_error":
                    failure_label, rollout_match_rate = apply_rollout_label_overrides(
                        raw_batch,
                        rollout_labels,
                        failure_label,
                    )
                else:
                    rollout_match_rate = 0.0
                horizon_label = horizon_labels(
                    pred_original,
                    target_original,
                    refinement_config.horizon_choices,
                    args.horizon_mse_threshold,
                )

            out = stack(obs_tokens, pred)
            losses = []
            logs: dict[str, float] = {}
            if "probe" in out:
                probe = out["probe"]
                probe_loss = F.binary_cross_entropy_with_logits(probe["failure_logit"], failure_label)
                losses.append(probe_loss)
                logs["probe_loss"] = float(probe_loss.detach().cpu())
            if "verifier" in out:
                verifier = out["verifier"]
                safety_label = 1.0 - failure_label
                verifier_loss = F.binary_cross_entropy_with_logits(verifier["safety_logit"], safety_label)
                advantage_target = -diagnostics["sample_mse"]
                advantage_loss = F.mse_loss(verifier["advantage"], advantage_target)
                combined = verifier_loss + 0.1 * advantage_loss
                losses.append(args.verifier_loss_weight * combined)
                logs["verifier_loss"] = float(combined.detach().cpu())
            if "horizon" in out:
                horizon = out["horizon"]
                horizon_loss = F.cross_entropy(horizon["horizon_logits"], horizon_label)
                losses.append(args.horizon_loss_weight * horizon_loss)
                logs["horizon_loss"] = float(horizon_loss.detach().cpu())
            if "recovery" in out:
                refined = out["recovery"]["refined_actions"]
                residual_steps = min(refined.shape[1], target.shape[1])
                residual_loss = F.mse_loss(refined[:, :residual_steps], target[:, :residual_steps, : refined.shape[-1]])
                losses.append(args.residual_loss_weight * residual_loss)
                logs["residual_loss"] = float(residual_loss.detach().cpu())

            if not losses:
                raise RuntimeError("No refinement heads are enabled")
            loss = sum(losses)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            if step % args.log_every == 0:
                print(
                    f"step={step} epoch={epoch} loss={float(loss.detach().cpu()):.6f} "
                    f"failure_rate={float(failure_label.mean().detach().cpu()):.3f} "
                    f"rollout_match_rate={rollout_match_rate:.3f} "
                    f"mse_original={float(diagnostics['sample_mse'].mean().detach().cpu()):.6f} "
                    f"smoothness_original={float(action_smoothness(pred_original).detach().cpu()):.6f} "
                    f"jerk_original={float(action_jerk(pred_original).detach().cpu()):.6f} "
                    f"jump_original={float(command_jump_ratio(pred_original, args.max_delta).detach().cpu()):.6f} "
                    f"logs={logs}",
                    flush=True,
                )
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(output_dir / f"step_{step:08d}.pt", stack, optimizer, args, refinement_config, step)
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    save_checkpoint(output_dir / "last.pt", stack, optimizer, args, refinement_config, step)
    print(f"saved_refinement_checkpoint={output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
