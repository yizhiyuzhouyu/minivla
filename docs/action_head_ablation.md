# Action Head Ablation Plan

MiniVLA keeps three action heads behind the same observation memory so the
comparison is about action generation rather than data loading or visual
encoding.

## Heads

| Head | Role | Config |
| --- | --- | --- |
| MLP | Simplest behavior cloning baseline. Pools observation memory and directly regresses an action chunk. | `action_head: mlp` |
| Query | ACT-style decoder baseline. Learnable action queries attend to observation memory and produce one action per future step. | `action_head: query` |
| Flow Matching | pi0-style mainline. A DiT-style Action Expert predicts a velocity field and samples action chunks by denoising. | `action_head: flow_matching` |

## Fixed Conditions

Keep these fixed across runs:

- Same LeRobot dataset and train/validation split.
- Same image keys, state dimension, action dimension, `chunk_size`, and
  `n_action_steps`.
- Same observation memory settings: visual encoder, token resampler,
  Transformer depth, hidden size, and tokenizer.
- Same normalizer stats.
- Same training steps, batch size, learning rate, weight decay, and seed.

Only change `action_head`.

## Metrics

Report the following metrics from `scripts/evaluate_policy.py`:

- `mean_fm_loss`: training objective loss. For MLP/Query this is direct action
  MSE under the same output name for easy table comparison.
- `mean_sampled_action_mse`: MSE between sampled/predicted chunks and target
  chunks.
- `mean_action_smoothness`: average `|a_t - a_{t-1}|`; lower usually means
  smoother commands.
- `mean_action_jerk`: average `|a_t - 2a_{t-1} + a_{t-2}|`; lower usually
  means fewer abrupt acceleration changes.
- `latency_ms`: average policy inference latency on evaluation batches.

## Commands

Train the three variants:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/train.py \
  --config configs/ablation_mlp.yaml \
  --dataset-repo-id your_org/your_lerobot_dataset

/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/train.py \
  --config configs/ablation_query.yaml \
  --dataset-repo-id your_org/your_lerobot_dataset

/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/train.py \
  --config configs/so101_pi0_like.yaml \
  --dataset-repo-id your_org/your_lerobot_dataset
```

Evaluate each checkpoint:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/evaluate_policy.py \
  --checkpoint outputs/ablation_mlp/last.pt \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --max-batches 20
```

Repeat with `outputs/ablation_query/last.pt` and
`outputs/so101_pi0_like/last.pt`.

## How To Interpret

- MLP is expected to be fastest and weakest on multimodal temporal structure.
- Query should be a stronger ACT-style baseline for chunk structure.
- Flow Matching is the highlighted pi0-style route; it may cost more inference
  time but gives a more expressive action distribution.

Do not claim a winner without running the same dataset and split. This document
defines the comparison protocol, not results.
