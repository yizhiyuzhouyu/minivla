# Post-SFT Policy Refinement Pipeline

MiniVLA should not treat SFT/BC as the final step. The post-SFT stage turns a
trained checkpoint into a deployable policy candidate through evaluation,
selection, calibration, action safety, failure mining, and future re-training.

The goal is not to claim pi0-scale post-training. The goal is to copy the
engineering discipline that matters for SO-101-scale experiments: restore the
same preprocessing path, choose checkpoints with validation metrics, tune
inference and execution parameters, record failures, and feed those failures
back into the next data or RL iteration.

## External Reference Mapping

MiniVLA follows these public pi0/openpi-style lessons at a smaller scale:

| Reference idea | MiniVLA mapping |
| --- | --- |
| openpi provides base/fine-tuned checkpoints, LeRobot conversion, norm stats, and policy server workflows | MiniVLA stores config, `norm_stats`, processor assets, and exposes `MiniVLAPolicyRunner`, `serve_policy.py`, and SO-101 client scripts |
| pi0.5 emphasizes heterogeneous co-training and high/low-level conditioning | MiniVLA keeps language tokens, optional `subtask.label`, `episode.*` metadata, multi-camera image tokens, and a separate action expert |
| pi0.5 uses continuous flow matching for low-level action chunks | MiniVLA mainline is `FMHead` with a DiT-style action expert and configurable denoise steps |
| pi*0.6 / RECAP-style post-SFT RL emphasizes correction, recovery, and learning from rollout failures | MiniVLA records high-loss, high-jerk, saturation, command-jump, and failure-case indices for recovery data collection and future RL |
| pi0.7-style context/data-quality improvements emphasize better labels and deployment metadata | MiniVLA already has hooks for `episode.success`, `episode.quality`, `subtask.label`, `future.image`, and configurable metadata tokens |

## Layer 1: Offline Evaluation And Checkpoint Selection

Do not deploy `last.pt` by default. Run every candidate checkpoint on the same
held-out LeRobot split and select by a scorecard.

For new runs, the same validation discipline can happen during SFT:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/train.py \
  --config configs/so101_pi0_like.yaml \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --split-json path/to/so101_train.json \
  --val-split-json configs/splits/so101_val.json \
  --eval-every 1000 \
  --best-metric selection_score
```

This saves `best.pt` from the training loop and writes `train_log.jsonl` plus
`train_log.csv`. Quality-aware SFT can additionally use `episode.success`,
`episode.quality`, `subtask.label`, and a previous `failure_cases.jsonl`:

```bash
  --use-quality-weights \
  --failure-manifest outputs/post_sft/failure_cases.jsonl \
  --loss-clip 2.0 \
  --fm-action-smoothness-loss-weight 0.01
```

Implemented entry point:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/post_sft_select_checkpoint.py \
  --checkpoint-glob "outputs/minivla/step_*.pt" \
  --checkpoints outputs/minivla/last.pt \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --output-dir outputs/post_sft
```

Outputs:

- `outputs/post_sft/report.json`
- `outputs/post_sft/best.pt`
- `outputs/post_sft/failure_cases.jsonl`

Scorecard metrics:

- `mean_fm_loss`
- `mean_sampled_action_mse` / `mean_sampled_action_mse_original`
- `mean_sampled_action_mse_normalized`
- `mean_action_smoothness`
- `mean_action_jerk`
- `latency_ms`
- `per_action_dim_mse`
- `per_episode`
- `action_saturation_ratio`
- `command_jump_ratio`

`mean_fm_loss` remains a normalized training-space objective. Action MSE,
smoothness, jerk, saturation, and command-jump metrics in the post-SFT
scorecard are computed after unnormalizing actions with checkpoint stats, so
they match deployment command scale.

Failure-case tags:

- `high_loss`
- `high_action_mse`
- `action_jump`
- `high_jerk`
- `action_saturation`
- `command_jump`

## Layer 2: Action Postprocess And Safety

Model outputs should not be sent directly to hardware. They pass through an
explicit postprocess layer:

- action unnormalization
- action dimension selection
- NaN/Inf guard
- clamp to command range
- max-delta rate limiting
- joint-limit projection
- EMA smoothing
- latency/frequency monitoring

Implemented module:

```text
src/minivla/postprocess.py
```

Main classes:

- `PostProcessConfig`
- `ActionPostProcessor`
- `LatencyMonitor`

The SO-101 client now uses this module instead of local one-off safety classes.
That makes the same postprocess path reusable by dry-runs, calibration scripts,
policy servers, and robot-side clients.

## Layer 3: Inference Parameter Calibration

Flow matching SFT usually needs deployment calibration. The key tuning target is
not just low MSE; it is a Pareto tradeoff across MSE, smoothness, jerk, command
saturation, jump rate, and latency.

Implemented entry point:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/post_sft_calibrate.py \
  --checkpoint outputs/post_sft/best.pt \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --output-dir outputs/post_sft
```

Default sweep:

- `num_inference_steps`: 2, 4, 8, 10, 16
- `n_action_steps`: 25
- `ema_alpha`: 0.2, 0.35, 0.5
- `max_delta`: 0.1, 0.2, 0.35
- `action_min/action_max`: -1.0 / 1.0
- `temporal_ensemble_decay`: 0.5

Outputs:

- `outputs/post_sft/calibration_report.json`
- `outputs/post_sft/recommended_post_sft_config.yaml`

Important limitation: offline calibration evaluates chunks on validation
observations. True temporal ensembling still needs closed-loop dry-run or robot
rollout validation because temporal ensemble behavior depends on consecutive
observations and execution timing.

Closed-loop validation is implemented as a replay script:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/replay_policy_sequence.py \
  --checkpoint outputs/post_sft/best.pt \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --split-json configs/splits/so101_val.json \
  --output-dir outputs/replay_val
```

Unlike static calibration, replay calls `select_action()`, so queued chunks,
temporal ensembling, and the stateful action postprocessor are all exercised.

## Layer 4: Trainable Runtime Refinement Heads

Postprocess heuristics are the first step. The next step is a trainable frozen
policy refinement stack:

```text
frozen MiniVLA action generator
  + ActionProbe(action_chunk)
  + ActionVerifierHead(obs_tokens, action_chunk)
  + AdaptiveHorizonHead(obs_tokens, action_chunk)
  + optional ResidualRecoveryPolicy(obs_tokens, action_chunk)
  -> ActionPostProcessor
```

Implemented modules:

- `ActionProbe`: action-only failure probability.
- `ActionVerifierHead`: observation-conditioned safety and advantage score.
- `AdaptiveHorizonHead`: dynamic execution horizon instead of fixed
  `n_action_steps`.
- `ResidualRecoveryPolicy`: bounded `delta_action` policy that leaves MiniVLA
  frozen.
- `PostSFTRefinementStack`: composable wrapper for ablation experiments.

Implemented training entry:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/train_refinement_heads.py \
  --checkpoint outputs/post_sft/best.pt \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --preset probe_verifier_horizon \
  --output-dir outputs/refinement_probe_verifier_horizon
```

See `docs/post_sft_refinement_ablation.md` for the staged ablation ladder.

Implemented evaluation entry:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/evaluate_refinement_heads.py \
  --checkpoint outputs/post_sft/best.pt \
  --refinement pvh=outputs/refinement_probe_verifier_horizon/last.pt \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --split-json configs/splits/so101_val.json \
  --output-dir outputs/refinement_ablation
```

The policy server can load the same refinement checkpoint:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/serve_policy.py \
  --checkpoint outputs/post_sft/best.pt \
  --refinement-checkpoint outputs/refinement_probe_verifier_horizon/last.pt
```

Default refinement labels are pseudo-labels derived from SFT prediction error.
That is useful for probes, failure mining, and ablation plumbing, but it is not
evidence of real policy improvement by itself. Treat a refinement head as a
deployment improvement only after validating it on held-out episodes with
rollout logs, human labels, or real robot success/failure outcomes.

## Layer 5: Failure Mining And Re-SFT / RL

Post-SFT should generate the next training agenda:

1. Run checkpoint selection and calibration.
2. Inspect `failure_cases.jsonl`.
3. Group failures into action, perception, language, and recovery buckets.
4. Collect targeted recovery demonstrations or label episode quality.
5. Re-run SFT with better weighting or filtering.
6. When real rollouts are available, add a constrained RL stage.

Recommended failure buckets:

| Bucket | Signal | Next action |
| --- | --- | --- |
| Action instability | high jerk, command jump | tune `max_delta`, EMA, temporal ensemble; add smooth recovery demos |
| Saturation | high clamp ratio | inspect normalization stats, action range, and joint limits |
| Per-action weakness | high `per_action_dim_mse` | inspect joint-specific control or gripper labels |
| Episode weakness | high per-episode loss | filter low-quality demos or upweight recovery demonstrations |
| Visual OOD | high loss in specific scenes | collect environment/object variants |
| Language mismatch | failure tied to task/subtask labels | improve `task` and `subtask.label` annotation |
| Recovery failure | policy cannot recover after perturbation | add reset/recovery demonstrations or RL rollouts |

## RL Complement

Do not jump directly from SFT to unconstrained online RL. A safer SO-101 path is:

1. **Offline failure mining:** identify high-risk states and actions from
   validation and dry-run logs.
2. **Recovery data collection:** collect short demonstrations from failed states
   back to valid task progress.
3. **Reweighted SFT:** upweight recovery clips and high-quality successful
   episodes; downweight noisy or failed demonstrations.
4. **Reward model or heuristic reward:** start with task success, action
   smoothness, joint-limit violations, and recovery progress.
5. **Constrained online RL:** initialize from SFT, limit action deltas and joint
   ranges, keep emergency stop active, and log every rollout for replay.

MiniVLA already has the minimal hooks for this path:

- `episode.success`
- `episode.quality`
- `subtask.label`
- `future.image`
- action clamp and rate limits
- failure-case JSONL output
- policy server / robot client split
- rollout JSONL logger
- explicit failure taxonomy config

The next useful RL implementation should be small: a rollout logger, a replay
buffer over failure/recovery snippets, and a reward function that penalizes
unsafe command changes before attempting any full online policy optimization.

## Interview Framing

Use this phrasing:

> After SFT, I do not directly deploy `last.pt`. I run a post-SFT policy
> refinement pipeline: checkpoint selection on a validation scorecard, action
> safety postprocess, FM inference calibration, and failure mining. The failure
> logs become the input to the next SFT or a constrained RL/recovery-data stage.
> This is inspired by openpi/pi0-style engineering: normalization assets and
> policy server discipline from openpi, heterogeneous context and action-expert
> separation from pi0.5, recovery/RL thinking from pi*0.6, and metadata/data
> quality conditioning from later pi0-style systems.

Do not overclaim:

- MiniVLA does not yet implement full pi0.6-scale RL.
- Offline smoothness and MSE are not replacements for real robot success rate.
- Temporal ensemble calibration from static validation data is only a proxy.
- Pseudo-labeled refinement heads are failure-analysis tools until validated
  against held-out rollout or human labels.
