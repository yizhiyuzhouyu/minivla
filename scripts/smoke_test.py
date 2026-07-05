import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minivla import MiniVLAConfig, MiniVLAPolicy
from minivla.constants import (
    ACTION,
    ACTION_IS_PAD,
    EPISODE_QUALITY,
    EPISODE_SUCCESS,
    FUTURE_IMAGE,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    SUBTASK_ATTENTION_MASK,
    SUBTASK_TOKENS,
)


def main() -> None:
    torch.manual_seed(0)
    cfg = MiniVLAConfig(
        use_hf_vision_encoder=False,
        image_keys=("observation.images.front", "observation.images.wrist"),
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
        num_inference_steps=2,
        future_latent_loss_weight=0.1,
    )
    policy = MiniVLAPolicy(cfg)

    batch = {
        "observation.images.front": torch.rand(2, 3, 64, 64),
        "observation.images.wrist": torch.rand(2, 3, 64, 64),
        OBS_LANGUAGE_TOKENS: torch.randint(0, cfg.text_vocab_size, (2, 12)),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 12, dtype=torch.bool),
        SUBTASK_TOKENS: torch.randint(0, cfg.text_vocab_size, (2, 6)),
        SUBTASK_ATTENTION_MASK: torch.ones(2, 6, dtype=torch.bool),
        OBS_STATE: torch.randn(2, 10),
        ACTION: torch.randn(2, cfg.chunk_size, cfg.action_dim),
        ACTION_IS_PAD: torch.zeros(2, cfg.chunk_size, dtype=torch.bool),
        EPISODE_SUCCESS: torch.tensor([1.0, 0.0]),
        EPISODE_QUALITY: torch.tensor([5.0, 2.0]),
        FUTURE_IMAGE: torch.rand(2, 3, 64, 64),
    }
    batch[ACTION_IS_PAD][0, -2:] = True

    loss, info = policy(batch)
    loss_none, _ = policy(batch, reduction="none")
    loss.backward()
    action_chunk = policy.predict_action_chunk({k: v for k, v in batch.items() if k != ACTION})
    single_action = policy.select_action({k: v for k, v in batch.items() if k != ACTION})
    obs_tokens = policy.encode_observation_tokens({k: v for k, v in batch.items() if k != ACTION})
    fm_actions = policy.fm_head.sample(obs_tokens, num_steps=1)

    assert torch.isfinite(loss)
    assert loss_none.shape[-1] == cfg.action_dim
    assert torch.all(loss_none[0, -2:] == 0)
    assert "future_latent_loss" in info
    assert action_chunk.shape == (2, cfg.chunk_size, cfg.action_dim)
    assert single_action.shape == (2, cfg.action_dim)
    assert fm_actions.shape == (2, cfg.chunk_size, cfg.max_action_dim)
    print(f"ok loss={float(info['loss']):.6f} action_chunk={tuple(action_chunk.shape)}")


if __name__ == "__main__":
    main()
