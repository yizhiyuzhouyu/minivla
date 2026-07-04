from __future__ import annotations

from collections import deque
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minivla.configuration_minivla import MiniVLAConfig
from minivla.constants import (
    ACTION,
    ACTION_IS_PAD,
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from minivla.fm_head import FMHead


def _torch_dtype(dtype: str) -> torch.dtype:
    return torch.bfloat16 if dtype == "bfloat16" else torch.float32


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    if vector.shape[-1] >= new_dim:
        return vector[..., :new_dim]
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def get_action_pad_mask(batch: dict[str, Tensor]) -> Tensor | None:
    for key in (ACTION_IS_PAD, "action_pad_mask", "actions_id_pad"):
        if key in batch:
            return batch[key]
    return None


class PatchVisionEncoder(nn.Module):
    """Small ViT-style fallback used when HF CLIP weights are not available."""

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.image_size = config.image_size
        self.patch_embed = nn.Conv2d(3, config.hidden_dim, kernel_size=config.patch_size, stride=config.patch_size)
        grid_h = config.image_size[0] // config.patch_size
        grid_w = config.image_size[1] // config.patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_h * grid_w, config.hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.hidden_dim * config.mlp_ratio),
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, config.num_dit_layers // 2))
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, images: Tensor) -> Tensor:
        if images.shape[-2:] != self.image_size:
            images = F.interpolate(images, size=self.image_size, mode="bilinear", align_corners=False)
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        return self.norm(self.encoder(tokens))


class HFVisionEncoder(nn.Module):
    """CLIP/ViT vision encoder plus a linear LLM projector."""

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel, CLIPVisionModel
        except ImportError as exc:
            raise ImportError("transformers is required when use_hf_vision_encoder=True") from exc

        hf_config = AutoConfig.from_pretrained(config.vision_model_name)
        if getattr(hf_config, "model_type", None) == "clip":
            self.backbone = CLIPVisionModel.from_pretrained(config.vision_model_name)
        else:
            self.backbone = AutoModel.from_pretrained(config.vision_model_name)
        hidden_size = self.backbone.config.hidden_size
        self.projector = nn.Linear(hidden_size, config.hidden_dim)
        self.image_size = config.image_size

        if config.freeze_vision_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images: Tensor) -> Tensor:
        if images.shape[-2:] != self.image_size:
            images = F.interpolate(images, size=self.image_size, mode="bilinear", align_corners=False)

        outputs = self.backbone(pixel_values=images)
        if hasattr(outputs, "last_hidden_state"):
            tokens = outputs.last_hidden_state
        else:
            raise RuntimeError("Unsupported HF vision model output")
        return self.projector(tokens)


class TextTokenEncoder(nn.Module):
    """Embedding + projection text encoder that preserves token-level language."""

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.text_vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)
        self.proj = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

    def forward(self, tokens: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        embedded = self.embedding(tokens)
        if mask is None:
            mask = torch.ones(tokens.shape, dtype=torch.bool, device=tokens.device)
        encoded = self.proj(embedded)
        return encoded, mask.to(dtype=torch.bool, device=encoded.device)


class StateEncoder(nn.Module):
    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(config.max_state_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.proj(state)[:, None, :]


class MiniVLAPolicy(nn.Module):
    """LeRobot-compatible MiniVLA policy."""

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = HFVisionEncoder(config) if config.use_hf_vision_encoder else PatchVisionEncoder(config)
        self.text_encoder = TextTokenEncoder(config)
        self.state_encoder = StateEncoder(config)
        self.camera_embedding = nn.Embedding(max(1, len(config.image_keys)), config.hidden_dim)
        self.image_modality = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.text_modality = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.state_modality = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.hidden_dim * config.mlp_ratio),
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.observation_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(1, config.num_dit_layers // 2),
        )
        self.observation_norm = nn.LayerNorm(config.hidden_dim)
        self.fm_head = FMHead(config)
        self.action_expert = self.fm_head.action_expert
        self._queues: dict[str, deque[Tensor]] = {}
        self.reset()
        self.to(dtype=_torch_dtype(config.dtype))
        if config.device is not None:
            self.to(config.device)

    def reset(self) -> None:
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def get_optim_params(self) -> Iterable[nn.Parameter]:
        return self.parameters()

    def sample_noise(self, shape: torch.Size | tuple[int, ...], device: torch.device) -> Tensor:
        return self.fm_head.sample_noise(shape, device, next(self.parameters()).dtype)

    def sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        return self.fm_head.sample_time(batch_size, device)

    def forward(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, Tensor | float]]:
        actions_raw = batch[ACTION]
        action_dim = min(actions_raw.shape[-1], self.config.max_action_dim)
        actions = pad_vector(actions_raw, self.config.max_action_dim).to(dtype=next(self.parameters()).dtype)
        obs_tokens = self.encode_observation_tokens(batch)
        return self.fm_head.loss(
            obs_tokens=obs_tokens,
            actions=actions,
            action_dim=action_dim,
            noise=noise,
            time=time,
            action_pad_mask=get_action_pad_mask(batch),
            reduction=reduction,
        )

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        return self.sample_actions(batch, noise=noise, num_steps=num_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        self.eval()
        obs_tokens = self.encode_observation_tokens(batch)
        actions = self.fm_head.sample(obs_tokens=obs_tokens, noise=noise, num_steps=num_steps)
        return actions[..., : self.config.action_dim]

    def encode_context(self, batch: dict[str, Tensor]) -> Tensor:
        return self.encode_observation_tokens(batch).mean(dim=1, keepdim=True)

    def encode_observation_tokens(self, batch: dict[str, Tensor]) -> Tensor:
        image_tokens = self.encode_images(batch)
        if self.config.max_image_tokens is not None and image_tokens.shape[1] > self.config.max_image_tokens:
            image_tokens = F.adaptive_avg_pool1d(
                image_tokens.transpose(1, 2),
                self.config.max_image_tokens,
            ).transpose(1, 2)
        image_tokens = image_tokens + self.image_modality

        text_tokens, text_mask = self.encode_text(batch)
        text_tokens = text_tokens + self.text_modality

        state_token = self.state_encoder(self.prepare_state(batch)) + self.state_modality

        obs_tokens = torch.cat([image_tokens, text_tokens, state_token], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros(
                    image_tokens.shape[:2],
                    dtype=torch.bool,
                    device=image_tokens.device,
                ),
                ~text_mask,
                torch.zeros(state_token.shape[:2], dtype=torch.bool, device=state_token.device),
            ],
            dim=1,
        )
        encoded = self.observation_encoder(obs_tokens, src_key_padding_mask=padding_mask)
        encoded = self.observation_norm(encoded)
        return encoded.masked_fill(padding_mask[:, :, None], 0.0)

    def encode_text(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        if OBS_LANGUAGE_TOKENS not in batch:
            raise KeyError(f"Missing {OBS_LANGUAGE_TOKENS}; use MiniVLAProcessor or provide tokenized text")
        lang_tokens = batch[OBS_LANGUAGE_TOKENS].long()
        lang_mask = batch.get(OBS_LANGUAGE_ATTENTION_MASK)
        return self.text_encoder(lang_tokens, lang_mask)

    def encode_images(self, batch: dict[str, Tensor]) -> Tensor:
        image_tensors = self.prepare_images(batch)
        encoded = []
        for index, image in enumerate(image_tensors):
            tokens = self.vision_encoder(image)
            camera_index = min(index, self.camera_embedding.num_embeddings - 1)
            tokens = tokens + self.camera_embedding.weight[camera_index][None, None, :]
            encoded.append(tokens)
        return torch.cat(encoded, dim=1)

    def prepare_images(self, batch: dict[str, Tensor]) -> list[Tensor]:
        keys = [key for key in self.config.image_keys if key in batch]
        if not keys:
            keys = [key for key in batch if key == OBS_IMAGE or key.startswith(f"{OBS_IMAGES}.")]
        if not keys:
            raise KeyError(f"No image tensor found. Expected one of {self.config.image_keys}")

        images = []
        for key in keys:
            image = batch[key]
            if image.ndim == 5:
                image = image[:, -1]
            if image.ndim != 4:
                raise ValueError(f"Image {key} must have shape (B,C,H,W) or (B,T,C,H,W), got {tuple(image.shape)}")
            images.append(image.to(dtype=next(self.parameters()).dtype))
        return images

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        if OBS_STATE not in batch:
            raise KeyError(f"Missing {OBS_STATE}")
        state = batch[OBS_STATE]
        if state.ndim == 3:
            state = state[:, -1]
        if state.ndim != 2:
            raise ValueError(f"{OBS_STATE} must have shape (B,D) or (B,T,D), got {tuple(state.shape)}")
        return pad_vector(state, self.config.max_state_dim).to(dtype=next(self.parameters()).dtype)
