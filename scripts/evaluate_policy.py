from __future__ import annotations

import argparse
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
from minivla.transforms import prepare_batch
from train import collate_batch


def load_lerobot_dataset(args: argparse.Namespace):
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise ImportError("Could not import LeRobotDataset. Install LeRobot first.") from exc

    kwargs: dict[str, Any] = {}
    if args.dataset_root is not None:
        kwargs["root"] = args.dataset_root
    if args.episodes is not None:
        kwargs["episodes"] = args.episodes
    return LeRobotDataset(args.dataset_repo_id, **kwargs)


def action_smoothness(actions: torch.Tensor) -> torch.Tensor:
    if actions.shape[1] < 2:
        return torch.zeros((), dtype=actions.dtype, device=actions.device)
    return (actions[:, 1:] - actions[:, :-1]).abs().mean()


def action_jerk(actions: torch.Tensor) -> torch.Tensor:
    if actions.shape[1] < 3:
        return torch.zeros((), dtype=actions.dtype, device=actions.device)
    return (actions[:, 2:] - 2.0 * actions[:, 1:-1] + actions[:, :-2]).abs().mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MiniVLA checkpoint loss on a LeRobot dataset.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--sample-actions", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    runner = MiniVLAPolicyRunner.from_checkpoint(args.checkpoint, device=args.device)
    dataset = load_lerobot_dataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=runner.device.type == "cuda",
        drop_last=False,
        collate_fn=collate_batch,
    )

    losses: list[float] = []
    action_mses: list[float] = []
    smoothness_values: list[float] = []
    jerk_values: list[float] = []
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
        with torch.no_grad():
            loss, _ = runner.policy(batch)
            losses.append(float(loss.detach().cpu()))
            if args.sample_actions:
                if runner.device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                pred = runner.policy.predict_action_chunk({k: v for k, v in batch.items() if k != ACTION})
                if runner.device.type == "cuda":
                    torch.cuda.synchronize()
                latencies_ms.append((time.perf_counter() - start) * 1000.0)
                target = batch[ACTION][..., : pred.shape[-1]]
                steps = min(pred.shape[1], target.shape[1])
                pred = pred[:, :steps]
                target = target[:, :steps]
                action_mses.append(float(torch.mean((pred - target) ** 2).detach().cpu()))
                smoothness_values.append(float(action_smoothness(pred).detach().cpu()))
                jerk_values.append(float(action_jerk(pred).detach().cpu()))

    if not losses:
        raise RuntimeError("No batches were evaluated")

    mean_loss = sum(losses) / len(losses)
    print(f"batches={len(losses)}")
    print(f"action_head={runner.policy.config.action_head}")
    print(f"mean_fm_loss={mean_loss:.6f}")
    if action_mses:
        mean_action_mse = sum(action_mses) / len(action_mses)
        mean_smoothness = sum(smoothness_values) / len(smoothness_values)
        mean_jerk = sum(jerk_values) / len(jerk_values)
        mean_latency_ms = sum(latencies_ms) / len(latencies_ms)
        print(f"sampled_batches={len(action_mses)}")
        print(f"mean_sampled_action_mse={mean_action_mse:.6f}")
        print(f"mean_action_smoothness={mean_smoothness:.6f}")
        print(f"mean_action_jerk={mean_jerk:.6f}")
        print(f"latency_ms={mean_latency_ms:.3f}")


if __name__ == "__main__":
    main()
