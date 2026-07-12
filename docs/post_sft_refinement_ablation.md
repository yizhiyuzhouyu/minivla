# Post-SFT Refinement Head Ablation

This ablation ladder upgrades MiniVLA from action postprocess heuristics to
trainable runtime refinement heads while keeping the base MiniVLA policy frozen.

## Module Ladder

| Stage | Preset | Enabled modules | Question |
| --- | --- | --- | --- |
| A0 | baseline | no trainable refinement heads | How good is frozen MiniVLA with static postprocess? |
| A1 | `probe` | `ActionProbe` | Can action chunks alone predict failure risk? |
| A2 | `probe_verifier` | `ActionProbe` + `ActionVerifierHead` | Does observation-conditioned chunk scoring improve filtering? |
| A3 | `probe_verifier_horizon` | A2 + `AdaptiveHorizonHead` | Can the system choose when to replan instead of fixed `n_action_steps`? |
| A4 | `full` | A3 + `ResidualRecoveryPolicy` | Can a small delta-action policy fix failed chunks without changing MiniVLA? |

## Implemented Modules

- `ActionProbe`: action-only sequence model. It consumes predicted chunks and
  predicts failure probability from magnitude, delta, jerk, saturation, jump,
  and consistency features.
- `ActionVerifierHead`: cross-modal verifier. It consumes observation memory
  tokens and a candidate action chunk, then predicts safety probability and an
  advantage-like score.
- `AdaptiveHorizonHead`: predicts the execution horizon from a discrete set
  such as `{1, 2, 4, 8, 16, 25}`.
- `ResidualRecoveryPolicy`: predicts bounded `delta_action` and returns
  `refined_actions`, leaving the base MiniVLA generator frozen.

Main code:

```text
src/minivla/refinement_heads.py
scripts/train_refinement_heads.py
scripts/evaluate_refinement_heads.py
```

## Pseudo-Labels Before Real Rollout Labels

Before real robot failure labels exist, train with pseudo-labels from the
frozen SFT policy:

- failure label: high sampled action MSE, smoothness, jerk, or jump ratio.
- verifier safety label: inverse of failure label.
- verifier advantage target: negative sampled action MSE.
- horizon label: longest prefix whose prefix MSE stays below a threshold.
- residual target: target action chunk minus frozen policy action chunk.

These labels are not a substitute for real rollout success/failure labels.
They are a practical bootstrapping step for module ablations.

Once rollout logs exist, pass them back into refinement training:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/train_refinement_heads.py \
  --checkpoint outputs/post_sft/best.pt \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --preset probe_verifier_horizon \
  --label-source rollout_human_labels \
  --rollout-jsonl outputs/rollouts/real_run.jsonl \
  --output-dir outputs/refinement_rollout_labels
```

Matched rollout labels override pseudo-labels by `(episode_id, frame_index)`.
Unmatched samples fall back to the existing prediction-error pseudo-labels.

## Train Commands

Train action-only probe:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/train_refinement_heads.py \
  --checkpoint outputs/post_sft/best.pt \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --preset probe \
  --output-dir outputs/refinement_probe
```

Train probe + verifier:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/train_refinement_heads.py \
  --checkpoint outputs/post_sft/best.pt \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --preset probe_verifier \
  --output-dir outputs/refinement_probe_verifier
```

Train probe + verifier + adaptive horizon:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/train_refinement_heads.py \
  --checkpoint outputs/post_sft/best.pt \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --preset probe_verifier_horizon \
  --output-dir outputs/refinement_probe_verifier_horizon
```

Train full stack with residual recovery:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/train_refinement_heads.py \
  --checkpoint outputs/post_sft/best.pt \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --preset full \
  --output-dir outputs/refinement_full
```

## Metrics

Report these metrics per preset:

- failure probe AUROC / accuracy once real labels exist.
- verifier safety BCE and correlation with rollout success.
- horizon accuracy against heuristic labels, then against real intervention
  points.
- residual refined action MSE.
- sampled action MSE before/after refinement.
- action smoothness and jerk before/after refinement.
- saturation ratio and command jump ratio.
- closed-loop success rate once SO-101 rollout labels exist.
- latency overhead from each added module.

Generate a machine-readable report and markdown table:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/evaluate_refinement_heads.py \
  --checkpoint outputs/post_sft/best.pt \
  --refinement probe=outputs/refinement_probe/last.pt \
  --refinement pvh=outputs/refinement_probe_verifier_horizon/last.pt \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --split-json configs/splits/so101_val.json \
  --output-dir outputs/refinement_ablation
```

Outputs:

- `outputs/refinement_ablation/report.json`
- `outputs/refinement_ablation/table.md`

## Deployment Rule

Use the modules conservatively:

1. Generate action chunk with frozen MiniVLA.
2. Score chunk with `ActionProbe`.
3. If verifier is enabled, score chunk with `ActionVerifierHead`.
4. If risk is high, resample or shorten horizon.
5. If residual recovery is enabled, apply bounded `delta_action`.
6. Pass the result through `ActionPostProcessor`.

This keeps failure handling outside the base model and makes every added module
ablatable.

## RL Extension

After real rollout labels exist, the same modules become the bridge to
constrained RL:

- `ActionProbe`: failure classifier / intervention trigger.
- `ActionVerifierHead`: reward or critic proxy.
- `AdaptiveHorizonHead`: dynamic replan policy.
- `ResidualRecoveryPolicy`: low-risk policy head for RL updates.

The first RL experiment should update only the residual recovery policy while
keeping MiniVLA frozen. Full-policy RL should be treated as a later experiment.

Rollout logs for that stage can be collected with:

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/log_rollout.py \
  --policy-url http://127.0.0.1:8010/infer \
  --observation-json path/to/so101_observation.json \
  --output outputs/rollouts/debug.jsonl \
  --dry-run
```
