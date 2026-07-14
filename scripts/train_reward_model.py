from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minivla import MiniVLAConfig, MiniVLAProcessor, MiniVLAPolicy
from minivla.policy import load_config
from minivla.rhf_sac import (
    RewardModelConfig,
    RolloutWindowDataset,
    TrajectoryRewardModel,
    collate_rollout_windows,
    split_rollout_batch,
)
from minivla.transforms import BatchNormalizer, prepare_batch


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_base(checkpoint_path: str | Path, device: torch.device) -> tuple[MiniVLAPolicy, MiniVLAProcessor, BatchNormalizer, MiniVLAConfig, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = load_config(checkpoint.get("config"))
    config.device = str(device)
    policy = MiniVLAPolicy(config).to(device)
    policy.load_compatible_state_dict(checkpoint["model"])
    policy.eval()
    for param in policy.parameters():
        param.requires_grad = False
    processor = MiniVLAProcessor(config)
    assets = checkpoint.get("assets") or {}
    normalizer_assets = assets.get("normalizer", {}) if isinstance(assets.get("normalizer"), dict) else {}
    normalizer = BatchNormalizer(
        checkpoint.get("norm_stats", checkpoint.get("dataset_stats")),
        device=device,
        normalize_state=bool(normalizer_assets.get("normalize_state", assets.get("normalize_state", True))),
        normalize_action=bool(normalizer_assets.get("normalize_action", assets.get("normalize_action", True))),
    )
    return policy, processor, normalizer, config, assets


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a MiniVLA RHF reward model from labeled rollout logs.")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--rollout-jsonl", nargs="+", required=True, help="Rollout jsonl files or directories.")
    parser.add_argument("--labels-jsonl", nargs="*", default=None, help="Optional labels keyed by episode_id/trajectory_id.")
    parser.add_argument("--output-dir", default="outputs/minivla_rhf_sac/reward_model")
    parser.add_argument("--device", default=_auto_device())
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    policy, processor, normalizer, config, assets = load_base(args.base_checkpoint, device)
    dataset = RolloutWindowDataset(
        args.rollout_jsonl,
        chunk_size=config.chunk_size,
        label_paths=args.labels_jsonl,
        require_labels=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_rollout_windows,
    )
    reward_model = TrajectoryRewardModel(
        config,
        RewardModelConfig(hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    log_path = output_dir / "reward_train_log.jsonl"
    step = 0
    reward_model.train()
    for epoch in range(args.epochs):
        for raw_batch in dataloader:
            obs_batch, _ = split_rollout_batch(raw_batch)
            batch = prepare_batch(obs_batch, config, processor, normalizer, device=device, require_action=True)
            with torch.no_grad():
                obs_tokens = policy.encode_observation_tokens(batch).float()
            outputs = reward_model(obs_tokens, batch["action"].float())
            loss, logs = reward_model.loss(outputs, raw_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1

            if step % args.log_every == 0:
                record = {"step": step, "epoch": epoch, **logs}
                write_jsonl(log_path, record)
                print(
                    f"step={step} epoch={epoch} loss={logs['loss']:.6f} reward_mse={logs['reward_mse']:.6f}",
                    flush=True,
                )

    reward_model.save_pretrained(output_dir, base_checkpoint=args.base_checkpoint)
    with (output_dir / "assets.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "base_checkpoint": args.base_checkpoint,
                "rollout_jsonl": args.rollout_jsonl,
                "labels_jsonl": args.labels_jsonl,
                "base_assets": assets,
            },
            handle,
            indent=2,
        )
    print(f"saved_reward_model={output_dir / 'reward_model.pt'}")


if __name__ == "__main__":
    main()
