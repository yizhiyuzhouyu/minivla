# MiniVLA Technical Report

## Motivation

MiniVLA was built to turn VLA papers and open-source systems into a small,
inspectable robot policy stack. The project focuses on SO-101-scale experiments
rather than full pi0-scale pretraining. The target is a compact loop:

1. Convert LeRobot-style demonstrations into normalized batches.
2. Encode image patch tokens, language tokens, and robot state into observation
   memory.
3. Generate action chunks with a configurable action head.
4. Restore the same preprocessing path at inference time.
5. Serve the policy and connect it to a robot-side safety client.

## From ACT To pi0

ACT is useful for understanding action chunking and query-based action
decoding. It addresses policy error accumulation by predicting a short horizon
of actions at once, and CVAE latent variables can model multiple demonstration
styles under the same observation.

pi0/openpi shifts the center of gravity toward a VLM-style observation prefix
and an action expert trained for continuous control. For MiniVLA, the practical
takeaway is:

- Keep visual information as patch/spatial tokens instead of one pooled vector.
- Fuse multimodal tokens into observation memory.
- Isolate continuous action generation in an action expert.
- Use flow matching for action chunks.
- Treat data normalization, checkpoint assets, serving, and robot execution as
  part of the model, not as afterthoughts.

## Current Architecture

MiniVLA currently contains:

- Multi-camera CLIP/ViT or fallback patch-ViT encoder.
- Learnable visual token resampler for fixed-size visual memory.
- Token-level language memory.
- State and optional metadata/subtask tokens.
- Observation Transformer.
- Configurable action heads:
  - `mlp`: direct pooled-memory behavior cloning baseline.
  - `query`: ACT-style action query decoder baseline.
  - `flow_matching`: DiT-style Flow Matching Action Expert mainline.
- Temporal action ensemble for overlapping chunks.
- Post-SFT policy refinement:
  - validation scorecard checkpoint selection.
  - reusable action safety postprocess.
  - trainable runtime refinement heads.
  - FM inference and execution-parameter calibration.
  - failure-case mining for re-SFT / recovery data / future RL.
- LeRobot-compatible training, checkpoint, evaluation, server, and SO-101
  client scaffold.

## Implemented Modules

- `src/minivla/modeling_minivla.py`: policy, observation memory, visual
  resampler, temporal ensemble.
- `src/minivla/fm_head.py`: DiT-style flow-matching action expert.
- `src/minivla/action_heads.py`: MLP and query-decoder baselines.
- `src/minivla/transforms.py`: LeRobot batch preparation and normalization.
- `src/minivla/postprocess.py`: action unnormalization hooks, action-dim
  selection, NaN/Inf guard, clamp, max-delta limiting, joint-limit projection,
  EMA smoothing, and latency monitoring.
- `src/minivla/refinement_heads.py`: `ActionProbe`, `ActionVerifierHead`,
  `AdaptiveHorizonHead`, `ResidualRecoveryPolicy`, and
  `PostSFTRefinementStack` for ablatable frozen-policy refinement.
- `src/minivla/policy.py`: checkpoint restore, `from_pretrained`,
  `save_pretrained`, `infer`, and `select_action`.
- `scripts/train.py`: YAML/CLI training entry.
- `scripts/post_sft_select_checkpoint.py`: validation scorecard checkpoint
  selection and failure-case logging.
- `scripts/post_sft_calibrate.py`: post-SFT FM denoise-step and action
  postprocess calibration.
- `scripts/train_refinement_heads.py`: trains post-SFT refinement heads from
  frozen MiniVLA predictions and pseudo-labels before real rollout labels are
  available.
- `scripts/evaluate_refinement_heads.py`: evaluates refinement ablations and
  writes `report.json` plus `table.md`.
- `scripts/log_rollout.py`: logs policy responses, postprocessed actions,
  refinement diagnostics, safety-filter info, and optional human labels for
  failure mining.
- `scripts/serve_policy.py`: HTTP policy server.
- `scripts/run_so101_policy.py`: robot-side safety client scaffold.
- `scripts/evaluate_policy.py`: offline loss evaluation.
- `scripts/benchmark_latency.py`: inference latency benchmark.
- `scripts/inspect_lerobot_dataset.py`: dataset schema inspection.

## Gap To pi0/openpi

MiniVLA is intentionally smaller:

- It does not train or fine-tune a full VLM backbone.
- It does not include internet-scale or cross-robot pretraining data.
- It has not yet reported verified real-robot success rates.
- The language path is lightweight token memory rather than full semantic
  reasoning.
- The SO-101 adapter still needs a concrete hardware API implementation.

## Next Experiments

The next useful experiments should avoid fake headline numbers and focus on
diagnostics:

- Compare `mlp`, `query`, and `flow_matching` action heads on the same dataset.
- Measure validation loss, sampled action MSE, action smoothness, and action
  jerk.
- Benchmark denoising steps versus latency.
- Compare adaptive pooling versus visual resampler.
- Compare queued action chunks versus temporal action ensemble.
- Run post-SFT checkpoint selection instead of deploying `last.pt` by default.
- Sweep `num_inference_steps`, `n_action_steps`, EMA, max-delta, action clamp,
  and temporal-ensemble settings to find an MSE/smoothness/jerk/latency Pareto
  point.
- Mine `failure_cases.jsonl` for high-loss, high-jerk, saturation, command
  jump, and per-episode failure patterns.
- Compare refinement ablations: action-only probe, probe + verifier, adaptive
  horizon, and residual recovery.
- Log dry-runs and real rollouts into JSONL, then replace pseudo-labels with
  rollout human labels or heuristic safety events.
- Use failure cases to collect recovery demonstrations, reweight SFT, and later
  add a constrained RL stage initialized from the SFT checkpoint.
- Add a small SO-101 real-robot task once data collection is available.

## Risks And Limits

- A small SO-101 dataset can overfit quickly; validation and failure-case
  logging matter more than adding modules.
- CVAE-style latent variables may collapse on single-mode demonstrations.
- Flow matching improves the action generation path, but it does not solve data
  coverage or out-of-distribution visual conditions by itself.
- Without real-robot evaluation, deployment claims should remain framed as
  engineering readiness rather than proven task performance.
