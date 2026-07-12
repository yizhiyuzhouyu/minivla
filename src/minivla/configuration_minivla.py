from __future__ import annotations

from dataclasses import dataclass, field

from minivla.constants import OBS_IMAGES


@dataclass
class MiniVLAConfig:
    """Configuration for MiniVLA.

    The defaults are conservative for local development. Set
    ``use_hf_vision_encoder=True`` to use a real Hugging Face CLIP/ViT backbone.
    """

    vlm_base_model_name: str = "lerobot/smolvla_base"
    vision_model_name: str = "openai/clip-vit-base-patch32"
    tokenizer_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    image_keys: tuple[str, ...] = (f"{OBS_IMAGES}.front",)
    image_size: tuple[int, int] = (224, 224)
    use_hf_vision_encoder: bool = True
    freeze_vision_encoder: bool = True
    patch_size: int = 16
    max_image_tokens: int | None = None
    image_token_reduction: str = "adaptive_pool"
    visual_resampler_layers: int = 1

    text_vocab_size: int = 49152
    tokenizer_max_length: int = 48
    pad_token_id: int = 0
    add_newline_to_task: bool = True

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50
    action_head: str = "flow_matching"
    max_state_dim: int = 32
    max_action_dim: int = 32
    action_dim: int | None = None
    use_temporal_ensemble: bool = False
    temporal_ensemble_max_chunks: int = 8
    temporal_ensemble_decay: float = 0.5
    use_episode_metadata: bool = False
    future_latent_loss_weight: float = 0.0

    hidden_dim: int = 512
    num_dit_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0
    fm_action_smoothness_loss_weight: float = 0.0

    device: str | None = None
    dtype: str = "float32"

    extra: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})"
            )
        if self.n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be positive")
        if self.max_state_dim <= 0 or self.max_action_dim <= 0:
            raise ValueError("max_state_dim and max_action_dim must be positive")
        if self.action_dim is None:
            self.action_dim = self.max_action_dim
        if self.action_dim <= 0 or self.action_dim > self.max_action_dim:
            raise ValueError("action_dim must be in (0, max_action_dim]")
        if self.action_head not in {"flow_matching", "mlp", "query"}:
            raise ValueError("action_head must be 'flow_matching', 'mlp', or 'query'")
        if self.future_latent_loss_weight < 0:
            raise ValueError("future_latent_loss_weight must be non-negative")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.image_size[0] % self.patch_size != 0 or self.image_size[1] % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.image_token_reduction not in {"adaptive_pool", "resampler"}:
            raise ValueError("image_token_reduction must be 'adaptive_pool' or 'resampler'")
        if self.visual_resampler_layers <= 0:
            raise ValueError("visual_resampler_layers must be positive")
        if self.temporal_ensemble_max_chunks <= 0:
            raise ValueError("temporal_ensemble_max_chunks must be positive")
        if self.temporal_ensemble_decay < 0:
            raise ValueError("temporal_ensemble_decay must be non-negative")
        if self.fm_action_smoothness_loss_weight < 0:
            raise ValueError("fm_action_smoothness_loss_weight must be non-negative")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be 'float32' or 'bfloat16'")
