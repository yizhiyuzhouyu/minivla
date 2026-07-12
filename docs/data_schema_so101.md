# SO-101 Data Schema

MiniVLA expects LeRobot-style batches. The training and inference paths share
the same preprocessing code in `src/minivla/transforms.py`.

## Required Keys

| Key | Shape | Type | Notes |
| --- | --- | --- | --- |
| `observation.images.front` | `[B, 3, H, W]` or `[B, T, 3, H, W]` | float or uint8 | Front camera. Additional cameras use `observation.images.*`; temporal windows are encoded when present. |
| `observation.state` | `[B, D]` or `[B, T, D]` | float | Robot proprioception. Temporal windows are flattened before padding. |
| `action` | `[B, chunk_size, D]` or `[B, D]` | float | Action chunk target during training. |
| `task` | `str` or `list[str]` | text | Used when language tokens are not precomputed. |

## Tokenized Language Keys

If language is pre-tokenized, `task` is not required.

| Key | Shape | Type |
| --- | --- | --- |
| `observation.language.tokens` | `[B, L]` | long |
| `observation.language.attention_mask` | `[B, L]` | bool |

## Optional Keys

| Key | Shape / Type | Purpose |
| --- | --- | --- |
| `action_is_pad` | `[B, chunk_size]` bool | Masks padded action steps from loss. |
| `episode.success` | `[B]` float/bool | Optional metadata token. |
| `episode.quality` | `[B]` float | Optional metadata token. |
| `subtask.label` | `str` or `list[str]` | Tokenized as additional language memory. |
| `future.image` | `[B, 3, H, W]` | Enables lightweight future latent auxiliary loss. |

## SO-101 Defaults

Typical SO-101 action/state settings should be adjusted to your robot driver:

- `action_dim: 7` for 6 arm joints plus gripper.
- `chunk_size: 50` at 30 Hz gives about 1.67 seconds of predicted actions.
- `n_action_steps: 25` executes half the chunk before refreshing.
- `n_obs_steps: 2` requests the current and previous observation from LeRobot;
  use `1` for datasets without observation history support.
- Use `observation.images.front` for single-camera runs.
- Use `observation.images.front` and `observation.images.wrist` for a front +
  wrist setup.

## Notes

- Images can be `uint8` in `[0, 255]`; preprocessing converts them to float.
  Hugging Face vision encoders then apply their pretrained image mean/std.
- State and action are normalized if dataset stats are available.
- `max_state_dim` and `max_action_dim` pad smaller robots into a fixed model
  shape while `action_dim` controls the active output dimensions.
- `future.image` only contributes the auxiliary future-latent loss when the
  dataset actually provides that key.
