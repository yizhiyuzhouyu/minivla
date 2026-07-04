from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minivla import MiniVLAConfig, MiniVLAProcessor, MiniVLAPolicy
from minivla.constants import ACTION, EPISODE_QUALITY, EPISODE_SUCCESS, FUTURE_IMAGE, SUBGOAL_IMAGE, SUBTASK_LABEL
from minivla.transforms import BatchNormalizer, prepare_batch


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


def load_lerobot_dataset(args: argparse.Namespace):
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise ImportError(
                "Could not import LeRobotDataset. Install LeRobot or pass a compatible import path."
            ) from exc

    kwargs: dict[str, Any] = {}
    if args.dataset_root is not None:
        kwargs["root"] = args.dataset_root
    if args.episodes is not None:
        kwargs["episodes"] = args.episodes

    if args.delta_timestamps:
        kwargs["delta_timestamps"] = {
            ACTION: [i / args.fps for i in range(args.chunk_size)],
        }

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
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_dit_layers": args.num_dit_layers,
        "chunk_size": args.chunk_size,
        "n_action_steps": args.n_action_steps,
        "max_state_dim": args.max_state_dim,
        "max_action_dim": args.max_action_dim,
        "action_dim": args.action_dim,
        "text_vocab_size": args.text_vocab_size,
        "tokenizer_max_length": args.tokenizer_max_length,
        "num_inference_steps": args.num_inference_steps,
        "max_image_tokens": args.max_image_tokens,
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
            "dataset_stats": stats,
            "assets": assets,
        },
        path,
    )


def load_checkpoint(path: str | None, map_location: str | torch.device) -> dict[str, Any] | None:
    if path is None:
        return None
    return torch.load(path, map_location=map_location)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MiniVLA on a LeRobot dataset.")
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot dataset repo id or local dataset id.")
    parser.add_argument("--dataset-root", default=None, help="Optional local dataset root.")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Optional episode ids to train on.")
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
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-dit-layers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=50)
    parser.add_argument("--max-state-dim", type=int, default=32)
    parser.add_argument("--max-action-dim", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--text-vocab-size", type=int, default=49152)
    parser.add_argument("--tokenizer-max-length", type=int, default=48)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--max-image-tokens", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
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

    policy = MiniVLAPolicy(config).to(device)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=args.lr, weight_decay=args.weight_decay)
    start_step = 0
    start_epoch = 0

    if checkpoint is not None:
        policy.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
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
        "normalize_state": args.normalize_state,
        "normalize_action": args.normalize_action,
        "processor": "MiniVLAProcessor",
        "tokenizer_name": config.tokenizer_name,
    }
    max_steps = args.max_steps if args.max_steps is not None else math.inf
    step = start_step

    policy.train()
    for epoch in range(start_epoch, args.epochs):
        for raw_batch in dataloader:
            batch = prepare_batch(raw_batch, config, processor, normalizer, device=device)
            loss, info = policy(batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip_norm)
            optimizer.step()

            step += 1
            if step % args.log_every == 0:
                print(f"step={step} epoch={epoch} loss={float(info['loss']):.6f}", flush=True)
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(output_dir / f"step_{step:08d}.pt", policy, optimizer, step, epoch, config, stats, assets)
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    save_checkpoint(output_dir / "last.pt", policy, optimizer, step, epoch, config, stats, assets)
    print(f"saved checkpoint: {output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
