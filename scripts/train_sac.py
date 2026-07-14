from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minivla import MiniVLAConfig, MiniVLAProcessor, MiniVLAPolicy
from minivla.policy import load_config
from minivla.rhf_sac import (
    RHFSACAgent,
    SACConfig,
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


@torch.no_grad()
def encode_and_base_actions(
    policy: MiniVLAPolicy,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    obs_tokens = policy.encode_observation_tokens(batch).float()
    obs_only = {key: value for key, value in batch.items() if key != "action"}
    base_actions = policy.predict_action_chunk(obs_only).float()
    return obs_tokens, base_actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a MiniVLA residual SAC policy using an RHF reward model.")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--reward-checkpoint", required=True)
    parser.add_argument("--rollout-jsonl", nargs="+", required=True)
    parser.add_argument("--labels-jsonl", nargs="*", default=None)
    parser.add_argument("--output-dir", default="outputs/minivla_rhf_sac/sac")
    parser.add_argument("--device", default=_auto_device())
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--actor-hidden-dim", type=int, default=None)
    parser.add_argument("--critic-hidden-dim", type=int, default=None)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--bc-weight", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device(args.device)
    policy, processor, normalizer, config, assets = load_base(args.base_checkpoint, device)
    reward_model = TrajectoryRewardModel.from_checkpoint(args.reward_checkpoint, config, map_location=device).to(device)
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False

    dataset = RolloutWindowDataset(
        args.rollout_jsonl,
        chunk_size=config.chunk_size,
        label_paths=args.labels_jsonl,
        require_labels=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_rollout_windows,
    )

    agent = RHFSACAgent(
        config,
        SACConfig(
            actor_hidden_dim=args.actor_hidden_dim,
            critic_hidden_dim=args.critic_hidden_dim,
            residual_scale=args.residual_scale,
            gamma=args.gamma,
            tau=args.tau,
            alpha=args.alpha,
            bc_weight=args.bc_weight,
        ),
    ).to(device)
    actor_optimizer = torch.optim.AdamW(agent.actor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    critic_optimizer = torch.optim.AdamW(
        list(agent.q1.parameters()) + list(agent.q2.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output_dir)
    log_path = output_dir / "sac_train_log.jsonl"
    step = 0
    for epoch in range(args.epochs):
        for raw_batch in dataloader:
            obs_raw, next_raw = split_rollout_batch(raw_batch)
            batch = prepare_batch(obs_raw, config, processor, normalizer, device=device, require_action=True)
            next_batch = prepare_batch(next_raw, config, processor, normalizer, device=device, require_action=False)
            done = raw_batch["done"].to(device=device, dtype=torch.float32)

            obs_tokens, base_actions = encode_and_base_actions(policy, batch)
            next_tokens, next_base_actions = encode_and_base_actions(policy, next_batch)
            data_actions = batch["action"].float()

            with torch.no_grad():
                reward = reward_model(obs_tokens, data_actions)["reward"].float()
                next_actions, next_log_prob, _ = agent.actor.sample(next_tokens, next_base_actions)
                target_q = torch.minimum(agent.target_q1(next_tokens, next_actions), agent.target_q2(next_tokens, next_actions))
                target = reward + args.gamma * (1.0 - done) * (target_q - args.alpha * next_log_prob)
                target = target.clamp(-agent.sac_config.q_clip, agent.sac_config.q_clip)

            q1 = agent.q1(obs_tokens, data_actions)
            q2 = agent.q2(obs_tokens, data_actions)
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_optimizer.step()

            actor_actions, log_prob, residual = agent.actor.sample(obs_tokens, base_actions)
            q_actor = torch.minimum(agent.q1(obs_tokens, actor_actions), agent.q2(obs_tokens, actor_actions))
            bc_loss = F.mse_loss(actor_actions, data_actions[..., : actor_actions.shape[-1]])
            actor_loss = (args.alpha * log_prob - q_actor).mean() + args.bc_weight * bc_loss
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_optimizer.step()
            agent.soft_update_targets()

            step += 1
            if step % args.log_every == 0:
                record = {
                    "step": step,
                    "epoch": epoch,
                    "critic_loss": float(critic_loss.detach().cpu()),
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "reward_mean": float(reward.mean().detach().cpu()),
                    "q_mean": float(q_actor.mean().detach().cpu()),
                    "residual_abs_mean": float(residual.abs().mean().detach().cpu()),
                    "bc_loss": float(bc_loss.detach().cpu()),
                }
                write_jsonl(log_path, record)
                print(
                    f"step={step} epoch={epoch} actor_loss={record['actor_loss']:.6f} "
                    f"critic_loss={record['critic_loss']:.6f} reward={record['reward_mean']:.4f}",
                    flush=True,
                )
            if args.save_every > 0 and step % args.save_every == 0:
                agent.save_pretrained(output_dir / f"step_{step:08d}", args.base_checkpoint, args.reward_checkpoint)

    agent.save_pretrained(output_dir, args.base_checkpoint, args.reward_checkpoint)
    with (output_dir / "assets.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "base_checkpoint": args.base_checkpoint,
                "reward_checkpoint": args.reward_checkpoint,
                "rollout_jsonl": args.rollout_jsonl,
                "labels_jsonl": args.labels_jsonl,
                "base_assets": assets,
            },
            handle,
            indent=2,
        )
    print(f"saved_sac={output_dir / 'sac.pt'}")


if __name__ == "__main__":
    main()
