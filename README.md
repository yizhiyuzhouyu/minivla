# MiniVLA-RHF-SAC

MiniVLA-RHF-SAC is a compact SO-101 VLA stack: MiniVLA base policy, SFT/BC,
rollout logging, trajectory-label reward modeling, and residual SAC
post-training.

MiniVLA is a compact LeRobot-compatible VLA policy project for SO-101 style
robot experiments. The goal is not to reproduce the full scale of pi0/openpi,
but to keep the same engineering shape at a smaller scale: multimodal
observation tokens, an isolated action expert, flow-matching action generation,
dataset normalization, checkpoint assets, and a policy server/client path.

The current mainline is a pi0-like lightweight policy:

- Vision: multi-camera CLIP/ViT patch tokens or an offline patch-ViT fallback;
  Hugging Face vision backbones use their pretrained image mean/std.
- Language: token-level task memory with attention masks, trained from scratch
  as a compact task embedding rather than a full pretrained language model.
- State: proprioception/state token from `observation.state`; temporal state
  windows are flattened when `n_obs_steps > 1`.
- Fusion: image/text/state/optional metadata tokens passed through an
  observation Transformer.
- Visual token control: adaptive pooling or a learnable query resampler for
  compressing dense patch tokens into a fixed visual memory.
- Action heads: configurable `mlp`, `query`, or `flow_matching` heads for
  ablation; the mainline uses a DiT-style Flow Matching Action Expert.
- Deployment smoothing: queued chunk execution or temporal action ensembling
  over overlapping chunks.
- Post-SFT refinement: validation scorecard checkpoint selection, action
  safety postprocess, trainable refinement heads, inference calibration, and
  failure-case mining for re-SFT/RL.
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
- Post-SFT action postprocess: `src/minivla/postprocess.py`
- Post-SFT refinement heads: `src/minivla/refinement_heads.py`
- Batch preprocessing: `src/minivla/transforms.py`
- Training entry: `scripts/train.py`
- Post-SFT checkpoint selection: `scripts/post_sft_select_checkpoint.py`
- Post-SFT calibration: `scripts/post_sft_calibrate.py`
- Post-SFT refinement head training: `scripts/train_refinement_heads.py`
- Post-SFT refinement evaluation: `scripts/evaluate_refinement_heads.py`
- RHF reward model training: `scripts/train_reward_model.py`
- RHF residual SAC training: `scripts/train_sac.py`
- Rollout logger: `scripts/log_rollout.py`
- Policy server: `scripts/serve_policy.py`
- Robot client scaffold: `scripts/run_so101_policy.py`

## pi0/openpi Correspondence

| pi0/openpi idea | MiniVLA implementation |
| --- | --- |
| VLM prefix / observation memory | Image, learned task-token, state, metadata tokens + observation Transformer |
| Visual token projector/resampler | `image_token_reduction=resampler` compresses dense patch tokens with learned queries |
| Action Expert separated from semantic memory | `FMHead` owns a DiT-style action expert |
| Flow Matching action generation | `x_t = t * noise + (1 - t) * action`, target velocity `noise - action` |
| Action head ablations | `action_head=mlp`, `query`, or `flow_matching` |
| Action chunking | `chunk_size` and `n_action_steps` config fields |
| Observation history | `n_obs_steps` requests multi-frame image/state windows from LeRobot |
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

Train with an in-loop validation split and quality-aware SFT options:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/train.py \
  --config configs/so101_pi0_like.yaml \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --split-json path/to/so101_train.json \
  --val-split-json configs/splits/so101_val.json \
  --eval-every 1000 \
  --use-quality-weights \
  --loss-clip 2.0 \
  --fm-action-smoothness-loss-weight 0.01
```

This writes `train_log.jsonl`, `train_log.csv`, `last.pt`, and validation-selected
`best.pt` under the output directory.

Training-time validation logs keep `mean_sampled_action_mse` as a normalized
training-space proxy and also write `mean_sampled_action_mse_original` after
unnormalizing actions with checkpoint stats. Post-SFT selection and calibration
use original action-space MSE, smoothness, jerk, saturation, and command-jump
metrics because those correspond to deployment commands.

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

## MiniVLA-RHF-SAC

`minivla-rhf-sac` keeps the base MiniVLA SFT path intact and adds a robot-data
post-training loop:

```text
MiniVLA base -> SFT/BC -> SO-101 rollout -> trajectory labels
             -> reward model -> residual SAC -> new rollouts
```

The rollout logger can now store the observation, action, video pointer, and
trajectory-level labels needed by RHF:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/log_rollout.py \
  --policy-url http://127.0.0.1:8010/infer \
  --output outputs/rollouts/task_a.jsonl \
  --episode-id task_a_0001 \
  --trajectory-id task_a_0001 \
  --save-observation \
  --success 1 \
  --stable-grasp 1 \
  --collision-free 1 \
  --smooth-action 1
```

For human review, labels can also be supplied later as JSONL records keyed by
`trajectory_id` or `episode_id`. Supported labels are `success`,
`stable_grasp`, `collision_free`, `smooth_action`, and `human_score`.

Train the reward model from labeled rollouts:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/train_reward_model.py \
  --base-checkpoint outputs/so101_pi0_like/best.pt \
  --rollout-jsonl outputs/rollouts \
  --labels-jsonl outputs/labels.jsonl \
  --output-dir outputs/minivla_rhf_sac/reward_model
```

Train residual SAC against the frozen base policy and learned reward:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/train_sac.py \
  --base-checkpoint outputs/so101_pi0_like/best.pt \
  --reward-checkpoint outputs/minivla_rhf_sac/reward_model/reward_model.pt \
  --rollout-jsonl outputs/rollouts \
  --output-dir outputs/minivla_rhf_sac/sac
```

Serve the SAC-refined policy:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/serve_policy.py \
  --checkpoint outputs/so101_pi0_like/best.pt \
  --sac-checkpoint outputs/minivla_rhf_sac/sac/sac.pt \
  --port 8010
```

The SAC layer is a bounded residual actor around MiniVLA's action chunk, with a
BC regularizer during training. This is intentional: it lets real SO-101 data
improve execution quality without immediately replacing the SFT policy.

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
mean_sampled_action_mse_normalized=
mean_sampled_action_mse_original=
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

Run post-SFT checkpoint selection:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/post_sft_select_checkpoint.py \
  --checkpoint-glob "outputs/minivla/step_*.pt" \
  --checkpoints outputs/minivla/last.pt \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --output-dir outputs/post_sft
```

Calibrate FM inference and action postprocess settings:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/post_sft_calibrate.py \
  --checkpoint outputs/post_sft/best.pt \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --output-dir outputs/post_sft
```

Replay a validation sequence through the same `select_action()` queue/temporal
ensemble path used at deployment:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/replay_policy_sequence.py \
  --checkpoint outputs/post_sft/best.pt \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --split-json configs/splits/so101_val.json \
  --output-dir outputs/replay_val
```

Train post-SFT refinement heads while keeping the base policy frozen:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/train_refinement_heads.py \
  --checkpoint outputs/post_sft/best.pt \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --preset probe_verifier_horizon \
  --output-dir outputs/refinement_probe_verifier_horizon
```

Evaluate refinement head ablations and write a report/table:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/evaluate_refinement_heads.py \
  --checkpoint outputs/post_sft/best.pt \
  --refinement probe=outputs/refinement_probe/last.pt \
  --refinement pvh=outputs/refinement_probe_verifier_horizon/last.pt \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --split-json configs/splits/so101_val.json \
  --output-dir outputs/refinement_ablation
```

Serve a policy with post-SFT refinement diagnostics:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/serve_policy.py \
  --checkpoint outputs/post_sft/best.pt \
  --refinement-checkpoint outputs/refinement_probe_verifier_horizon/last.pt \
  --port 8010
```

Log rollout or dry-run steps for failure mining:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/log_rollout.py \
  --policy-url http://127.0.0.1:8010/infer \
  --observation-json path/to/so101_observation.json \
  --output outputs/rollouts/debug.jsonl \
  --dry-run
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

When `delta_timestamps` is enabled, training and evaluation request the full
action chunk plus `n_obs_steps` image/state history from LeRobot. The default
SO-101 presets use `n_obs_steps: 2`; override with `--n-obs-steps 1` if a
dataset only supports single-frame observations.

`future.image` is optional. The `future_latent_loss_weight` setting only has an
effect when that key exists in the prepared batch.

## Current Scope

Implemented:

- LeRobot-compatible batch path.
- Multi-camera observation memory.
- Multi-frame observation history through `n_obs_steps`.
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
- Post-SFT checkpoint selection, action postprocess, calibration, and failure
  mining.
- Trainable post-SFT refinement heads: action probe, observation-conditioned
  verifier, adaptive horizon predictor, and optional residual recovery policy.
- Refinement ablation report generation and rollout logging for future
  recovery-data or constrained-RL loops.
- SO-101 safety client scaffold.

Not claimed yet:

- Full pi0-scale VLM pretraining.
- Pretrained LLM/VLM language understanding; language tokens are learned task
  embeddings in this compact stack.
- Verified cross-task generalization.
- Published real-robot success rates.
- Evidence that pseudo-labeled refinement heads improve real rollout success
  without held-out rollout or human-label validation.
- Production-ready SO-101 hardware driver.
