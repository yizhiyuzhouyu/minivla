from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minivla import MiniVLAConfig, MiniVLAPolicy
from minivla.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def main() -> None:
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
    )
    policy = MiniVLAPolicy(cfg)
    batch = {
        "observation.images.front": torch.rand(2, 3, 64, 64),
        OBS_LANGUAGE_TOKENS: torch.randint(0, cfg.text_vocab_size, (2, cfg.tokenizer_max_length)),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, cfg.tokenizer_max_length, dtype=torch.bool),
        OBS_STATE: torch.randn(2, cfg.max_state_dim),
        ACTION: torch.randn(2, cfg.chunk_size, cfg.action_dim),
    }
    loss, _ = policy(batch)
    actions = policy.predict_action_chunk({key: value for key, value in batch.items() if key != ACTION})
    print(f"loss={float(loss.detach()):.6f} actions_shape={tuple(actions.shape)}")


if __name__ == "__main__":
    main()
