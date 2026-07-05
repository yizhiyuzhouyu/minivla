# MiniVLA

MiniVLA is a pi0-style compact VLA stack for SO-101: patch-token observation
memory + Flow Matching Action Expert + LeRobot training/deployment loop.

MiniVLA is a compact LeRobot-compatible VLA policy project for SO-101 style
robot experiments. The goal is not to reproduce the full scale of pi0/openpi,
but to keep the same engineering shape at a smaller scale: multimodal
observation tokens, an isolated action expert, flow-matching action generation,
dataset normalization, checkpoint assets, and a policy server/client path.

The current mainline is a pi0-like lightweight policy:

- Vision: multi-camera CLIP/ViT patch tokens or an offline patch-ViT fallback.
- Language: token-level task memory with attention masks.
- State: proprioception/state token from `observation.state`.
- Fusion: image/text/state/optional metadata tokens passed through an
  observation Transformer.
- Visual token control: adaptive pooling or a learnable query resampler for
  compressing dense patch tokens into a fixed visual memory.
- Action heads: configurable `mlp`, `query`, or `flow_matching` heads for
  ablation; the mainline uses a DiT-style Flow Matching Action Expert.
- Deployment smoothing: queued chunk execution or temporal action ensembling
  over overlapping chunks.
- Tooling: LeRobot dataset entry point, checkpoint restore, HTTP policy server,
  evaluation, latency benchmark, data inspection, and SO-101 client scaffold.

## Why This Project

This repository is structured as a small but complete VLA engineering stack:

1. Train on LeRobot-style data.
2. Save config, normalizer statistics, and processor metadata into checkpoints.
3. Restore the exact preprocessing path for inference.
4. Serve a checkpoint through a simple policy API.
5. Run robot-side safety filtering before sending actions to hardware.

That mirrors the practical pieces that matter in pi0/openpi-style systems:
data format discipline, action expert separation, flow-matching control, and
deployment/evaluation plumbing.

## Architecture

```text
LeRobot batch
  observation.images.*  -> vision encoder -> camera-tagged image tokens
  observation.language  -> text encoder   -> language memory tokens
  observation.state     -> state encoder  -> proprioception token
  optional metadata     -> small encoders -> metadata/subtask tokens
                                                |
                                                v
                                  Observation Transformer memory
                                                |
                                                v
                          DiT Action Expert + Flow Matching Head
                                                |
                                                v
                                  action chunk [B, T, action_dim]
```

Key code paths:

- Policy model: `src/minivla/modeling_minivla.py`
- Flow-matching head: `src/minivla/fm_head.py`
- Action head baselines: `src/minivla/action_heads.py`
- Batch preprocessing: `src/minivla/transforms.py`
- Training entry: `scripts/train.py`
- Policy server: `scripts/serve_policy.py`
- Robot client scaffold: `scripts/run_so101_policy.py`

## pi0/openpi Correspondence

| pi0/openpi idea | MiniVLA implementation |
| --- | --- |
| VLM prefix / observation memory | Image, language, state, metadata tokens + observation Transformer |
| Visual token projector/resampler | `image_token_reduction=resampler` compresses dense patch tokens with learned queries |
| Action Expert separated from semantic memory | `FMHead` owns a DiT-style action expert |
| Flow Matching action generation | `x_t = t * noise + (1 - t) * action`, target velocity `noise - action` |
| Action head ablations | `action_head=mlp`, `query`, or `flow_matching` |
| Action chunking | `chunk_size` and `n_action_steps` config fields |
| Overlapping chunk smoothing | `use_temporal_ensemble` averages aligned chunk predictions |
| Dataset stats as deployment asset | Checkpoint stores `norm_stats` and normalizer metadata |
| Policy server / robot client separation | `serve_policy.py` + `run_so101_policy.py` |
| Small-model local iteration | `debug_tiny.yaml` and patch-ViT fallback |

MiniVLA deliberately does not claim full pi0-scale pretraining. It is a
research/engineering replica of the core control stack at SO-101 scale.

## Quick Start

Run the offline smoke test:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/smoke_test.py
```

Train from a YAML preset:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/train.py \
  --config configs/so101_front_wrist.yaml \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --output-dir outputs/minivla
```

Command-line flags override YAML values:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/train.py \
  --config configs/debug_tiny.yaml \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --max-steps 200 \
  --batch-size 4
```

Resume from a checkpoint:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/train.py \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --resume outputs/minivla/last.pt \
  --use-checkpoint-config
```

## Inference

Load a checkpoint in Python:

```python
from minivla import MiniVLAPolicyRunner

runner = MiniVLAPolicyRunner.from_checkpoint("outputs/minivla/last.pt")
actions = runner.infer(observation)["actions"]
```

Use a LeRobot-style pretrained directory:

```python
runner.save_pretrained("outputs/minivla_pretrained")
runner = MiniVLAPolicyRunner.from_pretrained("outputs/minivla_pretrained")
action = runner.select_action(observation)["action"]
```

Start the HTTP policy server:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/serve_policy.py \
  --checkpoint outputs/minivla/last.pt \
  --port 8010
```

Run the SO-101 client scaffold in dry-run mode:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/run_so101_policy.py \
  --policy-url http://127.0.0.1:8010/infer \
  --observation-json path/to/so101_observation.json \
  --dry-run
```

`run_so101_policy.py` includes action clamp, joint-limit checking, EMA smoothing,
emergency-stop file polling, and frequency/latency logging. The hardware adapter
is intentionally explicit: real SO-101 read/write methods should be connected
there instead of being hidden in the policy code.

To let the server perform policy-side chunk queueing or temporal ensembling,
start the same server and pass:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/run_so101_policy.py \
  --policy-url http://127.0.0.1:8010/infer \
  --observation-json path/to/so101_observation.json \
  --server-select-action \
  --dry-run
```

## Evaluation Tools

Inspect a LeRobot dataset:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/inspect_lerobot_dataset.py \
  --dataset-repo-id your_org/your_lerobot_dataset
```

Evaluate checkpoint loss on held-out batches:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/evaluate_policy.py \
  --checkpoint outputs/minivla/last.pt \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --max-batches 20
```

This prints action-head-neutral diagnostics:

```text
mean_fm_loss=
mean_sampled_action_mse=
mean_action_smoothness=
mean_action_jerk=
latency_ms=
```

Benchmark inference latency:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/benchmark_latency.py \
  --checkpoint outputs/minivla/last.pt \
  --warmup 10 \
  --iters 100
```

Plot generated or recorded action chunks:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/plot_action_chunks.py \
  --actions outputs/action_chunk.pt \
  --output outputs/action_chunk.png
```

Run the local project check:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/check_project.py
```

## Config Presets

- `configs/debug_tiny.yaml`: fast CPU/GPU sanity check.
- `configs/so101_front.yaml`: single front-camera SO-101 setup.
- `configs/so101_front_wrist.yaml`: front + wrist camera setup.
- `configs/so101_pi0_like.yaml`: larger pi0-like MiniVLA preset with frozen
  pretrained vision encoder, learnable visual resampler, FM action expert, and
  temporal action ensemble enabled.
- `configs/ablation_mlp.yaml`: direct pooled-memory MLP action baseline.
- `configs/ablation_query.yaml`: ACT-style action query decoder baseline.

## Batch Schema

Training data follows LeRobot-style names:

- Images: `observation.images.front`, `observation.images.wrist`, ...
- State: `observation.state`
- Action chunk: `action`
- Language tokens: `observation.language.tokens`
- Language mask: `observation.language.attention_mask`
- Optional task string: `task`
- Optional action padding: `action_is_pad`
- Optional metadata: `episode.success`, `episode.quality`, `subtask.label`,
  `future.image`

`src/minivla/transforms.py` handles key repacking, image conversion, resize,
normalization, action/state padding, and tokenization.

## Current Scope

Implemented:

- LeRobot-compatible batch path.
- Multi-camera observation memory.
- Learnable visual token resampler.
- Configurable MLP, query-decoder, and flow-matching action heads.
- Flow-matching DiT action expert as the mainline.
- Temporal action ensemble for overlapping chunks.
- Masked action loss.
- Checkpoint save/load with normalizer metadata.
- `save_pretrained` / `from_pretrained` runner API.
- Generated `model_card.md` for saved pretrained directories.
- Local project check script.
- HTTP policy server.
- Config presets.
- Dataset inspection, evaluation, latency, and plotting utilities.
- SO-101 safety client scaffold.

Not claimed yet:

- Full pi0-scale VLM pretraining.
- Verified cross-task generalization.
- Published real-robot success rates.
- Production-ready SO-101 hardware driver.
