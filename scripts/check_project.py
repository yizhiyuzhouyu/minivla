from __future__ import annotations

import compileall
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minivla import MiniVLAConfig, MiniVLAPolicy, PostSFTRefinementStack, RefinementConfig
from minivla.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def run_compileall() -> None:
    ok = True
    for relpath in ("scripts", "src", "examples"):
        ok = compileall.compile_dir(str(ROOT / relpath), quiet=1) and ok
    if not ok:
        raise RuntimeError("compileall failed")
    print("compileall=ok")


def run_smoke_test() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_test.py")], check=True)
    print("smoke_test=ok")


def config_kwargs(path: Path) -> dict[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    valid = MiniVLAConfig.__dataclass_fields__.keys()
    return {key: value for key, value in data.items() if key in valid and value is not None}


def instantiate_configs() -> None:
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        kwargs = config_kwargs(path)
        cfg = MiniVLAConfig(**kwargs)
        cfg.device = "cpu"
        if cfg.use_hf_vision_encoder:
            cfg.use_hf_vision_encoder = False
            cfg.dtype = "float32"
            cfg.hidden_dim = 64
            cfg.num_heads = 4
            cfg.num_dit_layers = 2
            cfg.image_size = (64, 64)
            cfg.max_image_tokens = 8
            cfg.text_vocab_size = 128
        _ = MiniVLAPolicy(cfg)
        print(f"config={path.name} action_head={cfg.action_head} ok")


def check_action_heads() -> None:
    for action_head in ("mlp", "query", "flow_matching"):
        cfg = MiniVLAConfig(
            use_hf_vision_encoder=False,
            image_keys=("observation.images.front",),
            image_size=(64, 64),
            patch_size=16,
            hidden_dim=64,
            num_heads=4,
            num_dit_layers=2,
            action_head=action_head,
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
        if not torch.isfinite(loss):
            raise RuntimeError(f"{action_head} loss is not finite")
        if actions.shape != (2, cfg.chunk_size, cfg.action_dim):
            raise RuntimeError(f"{action_head} returned wrong shape: {tuple(actions.shape)}")
        print(f"action_head={action_head} ok")


def check_refinement_heads() -> None:
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
    obs_tokens = policy.encode_observation_tokens({key: value for key, value in batch.items() if key != ACTION})
    actions = policy.predict_action_chunk({key: value for key, value in batch.items() if key != ACTION})
    stack = PostSFTRefinementStack(
        cfg,
        RefinementConfig(
            enable_action_probe=True,
            enable_verifier=True,
            enable_adaptive_horizon=True,
            enable_residual_recovery=True,
            horizon_choices=(1, 2, 4),
            residual_scale=0.1,
        ),
    )
    out = stack(obs_tokens, actions)
    if out["probe"]["failure_probability"].shape != (2,):
        raise RuntimeError("ActionProbe returned wrong shape")
    if out["verifier"]["safety_probability"].shape != (2,):
        raise RuntimeError("ActionVerifierHead returned wrong shape")
    if out["horizon"]["horizon"].shape != (2,):
        raise RuntimeError("AdaptiveHorizonHead returned wrong shape")
    if out["refined_actions"].shape != (2, cfg.chunk_size, cfg.action_dim):
        raise RuntimeError("ResidualRecoveryPolicy returned wrong shape")
    with tempfile.TemporaryDirectory() as tmpdir:
        stack.save_pretrained(tmpdir)
        restored = PostSFTRefinementStack.from_checkpoint(Path(tmpdir) / "refinement.pt", cfg)
        restored_out = restored(obs_tokens, actions)
        if restored_out["refined_actions"].shape != (2, cfg.chunk_size, cfg.action_dim):
            raise RuntimeError("Restored refinement stack returned wrong shape")
    print("refinement_heads=ok")


def main() -> None:
    run_compileall()
    instantiate_configs()
    check_action_heads()
    check_refinement_heads()
    run_smoke_test()
    print("project_check=ok")


if __name__ == "__main__":
    main()
