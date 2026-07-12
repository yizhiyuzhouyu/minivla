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

from minivla import MiniVLAPolicyRunner
from minivla.constants import ACTION
from minivla.postprocess import ActionPostProcessor, PostProcessConfig, action_jerk, action_smoothness
from minivla.splits import load_episode_split
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


def sample_value(raw_batch: dict[str, Any], key: str, default: Any = None) -> Any:
    value = raw_batch.get(key)
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        item = value[0]
        return item.item() if item.numel() == 1 else item.detach().cpu().tolist()
    if isinstance(value, list):
        return value[0] if value else default
    return value


def episode_id(raw_batch: dict[str, Any]) -> str:
    for key in ("episode_index", "episode.id", "episode", "episode_id"):
        value = sample_value(raw_batch, key)
        if value is not None:
            return str(value)
    return "unknown"


def first_target_action(raw_batch: dict[str, Any], action_dim: int | None = None) -> torch.Tensor:
    action = raw_batch[ACTION]
    if not torch.is_tensor(action):
        action = torch.as_tensor(action, dtype=torch.float32)
    action = action.float()
    if action.ndim == 3:
        target = action[0, 0]
    elif action.ndim == 2:
        target = action[0]
    elif action.ndim == 1:
        target = action
    else:
        raise ValueError(f"Expected action with 1-3 dims, got {tuple(action.shape)}")
    if action_dim is not None:
        target = target[:action_dim]
    return target


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a LeRobot sequence through MiniVLA select_action and postprocess.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--refinement-checkpoint", default=None)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--split-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/replay")
    parser.add_argument("--delta-timestamps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--action-min", type=float, default=-1.0)
    parser.add_argument("--action-max", type=float, default=1.0)
    parser.add_argument("--max-delta", type=float, default=0.2)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--saturation-eps", type=float, default=1e-6)
    args = parser.parse_args()

    runner = MiniVLAPolicyRunner.from_checkpoint(
        args.checkpoint,
        device=args.device,
        refinement_checkpoint=args.refinement_checkpoint,
    )
    dataset = load_lerobot_dataset(args, runner)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_batch,
    )
    postprocessor = ActionPostProcessor(
        PostProcessConfig(
            action_dim=runner.policy.config.action_dim,
            action_min=args.action_min,
            action_max=args.action_max,
            max_delta=args.max_delta,
            ema_alpha=args.ema_alpha,
            saturation_eps=args.saturation_eps,
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "replay.jsonl"
    current_episode: str | None = None
    executed_actions: list[torch.Tensor] = []
    target_actions: list[torch.Tensor] = []
    mse_values: list[float] = []
    latency_values: list[float] = []
    saturation_values: list[float] = []
    jump_values: list[float] = []

    with records_path.open("w", encoding="utf-8") as handle:
        for step, raw_batch in enumerate(dataloader):
            if step >= args.max_steps:
                break
            ep_id = episode_id(raw_batch)
            if ep_id != current_episode:
                current_episode = ep_id
                runner.policy.reset()
                postprocessor.reset()

            observation = {key: value for key, value in raw_batch.items() if key != ACTION}
            if runner.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = runner.select_action(observation)
            if runner.device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - start) * 1000.0

            raw_action = result["action"][0].detach().cpu().float()
            executed, info = postprocessor(raw_action)
            target = first_target_action(raw_batch, runner.policy.config.action_dim).cpu()
            dims = min(executed.numel(), target.numel())
            executed = executed[:dims]
            target = target[:dims]
            mse_value = float(torch.mean((executed - target) ** 2).item())

            executed_actions.append(executed)
            target_actions.append(target)
            mse_values.append(mse_value)
            latency_values.append(latency_ms)
            saturation_values.append(info.saturation_ratio)
            jump_values.append(info.command_jump_ratio)

            record = {
                "step": step,
                "episode_id": ep_id,
                "frame_index": sample_value(raw_batch, "frame_index"),
                "timestamp": sample_value(raw_batch, "timestamp"),
                "mse": mse_value,
                "latency_ms": latency_ms,
                "raw_action": raw_action.tolist(),
                "executed_action": executed.tolist(),
                "target_action": target.tolist(),
                "postprocess": info.to_dict(),
            }
            for key in ("failure_probability", "safety_probability", "advantage", "horizon", "expected_horizon"):
                if key in result:
                    value = result[key]
                    record[key] = value.detach().cpu().tolist() if torch.is_tensor(value) else value
            handle.write(json.dumps(record) + "\n")

    if not mse_values:
        raise RuntimeError("No replay steps were evaluated")

    executed_tensor = torch.stack(executed_actions, dim=0)[:, None, :]
    target_tensor = torch.stack(target_actions, dim=0)[:, None, :]
    report = {
        "checkpoint": args.checkpoint,
        "refinement_checkpoint": args.refinement_checkpoint,
        "steps": len(mse_values),
        "mean_executed_action_mse": mean(mse_values),
        "executed_action_smoothness": float(action_smoothness(executed_tensor.transpose(0, 1)).item()),
        "executed_action_jerk": float(action_jerk(executed_tensor.transpose(0, 1)).item()),
        "target_action_smoothness": float(action_smoothness(target_tensor.transpose(0, 1)).item()),
        "target_action_jerk": float(action_jerk(target_tensor.transpose(0, 1)).item()),
        "mean_latency_ms": mean(latency_values),
        "mean_saturation_ratio": mean(saturation_values),
        "mean_command_jump_ratio": mean(jump_values),
        "records": str(records_path),
    }
    with (output_dir / "replay_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"saved_records={records_path}")
    print(f"saved_report={output_dir / 'replay_report.json'}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
