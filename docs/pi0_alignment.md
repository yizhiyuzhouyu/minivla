# pi0/openpi Alignment

MiniVLA is not a full pi0 reproduction. It is a compact SO-101-scale project
that mirrors the practical architecture shape of pi0/openpi: patch-token
observation memory, an action expert, flow-matching action generation, and a
LeRobot training/deployment loop.

| pi0/openpi | MiniVLA |
| --- | --- |
| PaliGemma/SigLIP image prefix | CLIP/patch-ViT tokens |
| VLM prefix memory | Observation memory from image/text/state/metadata tokens |
| Visual projector/resampler | Linear projector plus optional learnable visual token resampler |
| Action Expert | DiT-style `FMHead` |
| Flow Matching | Implemented as the main action head |
| Action head alternatives | `mlp` and `query` baselines for ablation |
| Action chunking | `chunk_size`, `n_action_steps`, and temporal action ensemble |
| LeRobot data path | LeRobot-style batch schema and dataset stats normalizer |
| Policy server | Implemented with `scripts/serve_policy.py` |
| Robot client | SO-101 safety scaffold in `scripts/run_so101_policy.py` |
| Large-scale pretraining | Not claimed; MiniVLA is scoped to SO-101-scale experimentation |

## What Is Intentionally Similar

- **Observation memory instead of flat features.** Image patch tokens, language
  tokens, state tokens, and optional metadata tokens are fused by a Transformer.
- **Action generation is isolated.** The flow-matching head owns a separate
  action expert, matching the idea that semantic memory and continuous control
  should not be entangled in one MLP.
- **Flow matching is the mainline.** Direct MLP and query-decoder heads are kept
  as baselines, while the highlighted path is DiT + flow matching.
- **Deployment is part of the project.** Checkpoint assets, normalizer stats,
  policy server restore, robot-side safety filtering, and action smoothing are
  first-class components.

## What Is Intentionally Smaller

- MiniVLA does not train a large VLM backbone.
- MiniVLA does not claim pi0-level cross-robot or open-world generalization.
- The current language encoder is lightweight token memory, not a full LLM.
- The robot client is a safe SO-101 adapter scaffold, not a production hardware
  driver.

## Why This Still Matters

For an internship project, the valuable part is the engineering reduction:
turn the pi0/openpi idea into a small stack that can be trained, restored,
served, inspected, benchmarked, and eventually connected to SO-101 hardware.
That makes the project defensible in interviews: every major architectural
claim has a corresponding code path.
