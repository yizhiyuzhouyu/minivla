from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minivla import MiniVLAConfig, MiniVLAPolicy, MiniVLAPolicyRunner
from minivla.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def synthetic_batch(policy: MiniVLAPolicy, batch_size: int) -> dict[str, torch.Tensor]:
    cfg = policy.config
    device = next(policy.parameters()).device
    dtype = next(policy.parameters()).dtype
    batch: dict[str, torch.Tensor] = {}
    for key in cfg.image_keys:
        batch[key] = torch.rand(batch_size, 3, *cfg.image_size, device=device, dtype=dtype)
    batch[OBS_LANGUAGE_TOKENS] = torch.ones(
        batch_size,
        cfg.tokenizer_max_length,
        device=device,
        dtype=torch.long,
    )
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(
        batch_size,
        cfg.tokenizer_max_length,
        device=device,
        dtype=torch.bool,
    )
    batch[OBS_STATE] = torch.zeros(batch_size, cfg.max_state_dim, device=device, dtype=dtype)
    return batch


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round((pct / 100.0) * (len(values) - 1))))
    return sorted(values)[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MiniVLA action chunk inference latency.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint is not None:
        runner = MiniVLAPolicyRunner.from_checkpoint(args.checkpoint, device=device)
        policy = runner.policy
    else:
        cfg = MiniVLAConfig(
            use_hf_vision_encoder=False,
            image_keys=("observation.images.front",),
            image_size=(64, 64),
            patch_size=16,
            hidden_dim=64,
            num_heads=4,
            num_dit_layers=2,
            chunk_size=8,
            n_action_steps=4,
            max_state_dim=10,
            max_action_dim=10,
            action_dim=7,
            text_vocab_size=128,
            tokenizer_max_length=16,
            num_inference_steps=2,
            device=device,
        )
        policy = MiniVLAPolicy(cfg)
    policy.eval()
    batch = synthetic_batch(policy, args.batch_size)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = policy.predict_action_chunk(batch, num_steps=args.num_steps)
        if torch.cuda.is_available() and next(policy.parameters()).is_cuda:
            torch.cuda.synchronize()
        latencies_ms: list[float] = []
        for _ in range(args.iters):
            start = time.perf_counter()
            _ = policy.predict_action_chunk(batch, num_steps=args.num_steps)
            if torch.cuda.is_available() and next(policy.parameters()).is_cuda:
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    print(f"device={device} batch_size={args.batch_size} iters={args.iters}")
    print(f"mean_ms={statistics.mean(latencies_ms):.3f}")
    print(f"median_ms={statistics.median(latencies_ms):.3f}")
    print(f"p95_ms={percentile(latencies_ms, 95):.3f}")
    print(f"hz={1000.0 / statistics.mean(latencies_ms):.2f}")


if __name__ == "__main__":
    main()
