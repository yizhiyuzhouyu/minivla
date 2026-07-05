from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


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


def describe_value(value: Any) -> str:
    if torch.is_tensor(value):
        return f"tensor shape={tuple(value.shape)} dtype={value.dtype}"
    if isinstance(value, str):
        return f"str len={len(value)} value={value[:80]!r}"
    if isinstance(value, (float, int, bool)):
        return f"{type(value).__name__} value={value}"
    if isinstance(value, dict):
        return f"dict keys={list(value.keys())}"
    if isinstance(value, list):
        return f"list len={len(value)}"
    return type(value).__name__


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a LeRobot dataset sample and metadata.")
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    dataset = load_lerobot_dataset(args)
    print(f"dataset={args.dataset_repo_id}")
    print(f"num_samples={len(dataset)}")

    meta = getattr(dataset, "meta", None)
    for source_name, source in (("dataset", dataset), ("meta", meta)):
        if source is None:
            continue
        stats = getattr(source, "stats", None)
        if isinstance(stats, dict):
            print(f"{source_name}.stats keys={list(stats.keys())}")
        features = getattr(source, "features", None)
        if features is not None:
            print(f"{source_name}.features={features}")
        fps = getattr(source, "fps", None)
        if fps is not None:
            print(f"{source_name}.fps={fps}")

    sample = dataset[args.index]
    print(f"sample_index={args.index}")
    for key in sorted(sample.keys()):
        print(f"  {key}: {describe_value(sample[key])}")


if __name__ == "__main__":
    main()
