from __future__ import annotations

from collections import deque
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minivla.action_heads import MLPActionHead, QueryActionHead
from minivla.configuration_minivla import MiniVLAConfig
from minivla.constants import (
    ACTION,
    ACTION_DIM,
    ACTION_IS_PAD,
    EPISODE_QUALITY,
    EPISODE_SUCCESS,
    FUTURE_IMAGE,
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    SUBTASK_ATTENTION_MASK,
    SUBTASK_TOKENS,
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
            from transformers import AutoConfig, AutoImageProcessor, AutoModel, CLIPVisionModel
        except ImportError as exc:
            raise ImportError("transformers is required when use_hf_vision_encoder=True") from exc

        hf_config = AutoConfig.from_pretrained(config.vision_model_name)
        processor = AutoImageProcessor.from_pretrained(config.vision_model_name)
        if getattr(hf_config, "model_type", None) == "clip":
            self.backbone = CLIPVisionModel.from_pretrained(config.vision_model_name)
        else:
            self.backbone = AutoModel.from_pretrained(config.vision_model_name)
        hidden_size = self.backbone.config.hidden_size
        self.projector = nn.Linear(hidden_size, config.hidden_dim)
        self.image_size = config.image_size
        image_mean = getattr(processor, "image_mean", None) or [0.48145466, 0.4578275, 0.40821073]
        image_std = getattr(processor, "image_std", None) or [0.26862954, 0.26130258, 0.27577711]
        self.register_buffer("image_mean", torch.tensor(image_mean, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(image_std, dtype=torch.float32).view(1, -1, 1, 1), persistent=False)

        if config.freeze_vision_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images: Tensor) -> Tensor:
        if images.shape[-2:] != self.image_size:
            images = F.interpolate(images, size=self.image_size, mode="bilinear", align_corners=False)
        images = images.to(dtype=torch.float32)
        image_mean = self.image_mean.to(device=images.device, dtype=torch.float32)
        image_std = self.image_std.to(device=images.device, dtype=torch.float32).clamp_min(1e-6)
        images = (images - image_mean) / image_std
        images = images.to(dtype=next(self.backbone.parameters()).dtype)

        outputs = self.backbone(pixel_values=images)
        if hasattr(outputs, "last_hidden_state"):
            tokens = outputs.last_hidden_state
        else:
            raise RuntimeError("Unsupported HF vision model output")
        return self.projector(tokens)


class VisualTokenResampler(nn.Module):
    """Learned query resampler for compressing dense visual patch tokens."""

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        if config.max_image_tokens is None:
            raise ValueError("VisualTokenResampler requires max_image_tokens")
        self.query = nn.Parameter(torch.zeros(1, config.max_image_tokens, config.hidden_dim))
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "query_norm": nn.LayerNorm(config.hidden_dim),
                        "context_norm": nn.LayerNorm(config.hidden_dim),
                        "attn": nn.MultiheadAttention(
                            config.hidden_dim,
                            config.num_heads,
                            dropout=config.dropout,
                            batch_first=True,
                        ),
                        "ffn_norm": nn.LayerNorm(config.hidden_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(config.hidden_dim, int(config.hidden_dim * config.mlp_ratio)),
                            nn.GELU(),
                            nn.Linear(int(config.hidden_dim * config.mlp_ratio), config.hidden_dim),
                        ),
                    }
                )
                for _ in range(config.visual_resampler_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        queries = self.query.expand(tokens.shape[0], -1, -1).to(dtype=tokens.dtype, device=tokens.device)
        context = tokens
        for layer in self.layers:
            attn_out, _ = layer["attn"](
                layer["query_norm"](queries),
                layer["context_norm"](context),
                layer["context_norm"](context),
                need_weights=False,
            )
            queries = queries + attn_out
            queries = queries + layer["ffn"](layer["ffn_norm"](queries))
        return self.out_norm(queries)


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
        self.visual_resampler = (
            VisualTokenResampler(config)
            if config.max_image_tokens is not None and config.image_token_reduction == "resampler"
            else None
        )
        self.text_encoder = TextTokenEncoder(config)
        self.state_encoder = StateEncoder(config)
        self.camera_embedding = nn.Embedding(max(1, len(config.image_keys)), config.hidden_dim)
        self.obs_temporal_embedding = nn.Embedding(max(1, config.n_obs_steps), config.hidden_dim)
        self.image_modality = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.text_modality = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.state_modality = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.metadata_modality = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.subtask_modality = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.metadata_encoder = nn.Sequential(
            nn.Linear(2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
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
        if config.action_head == "flow_matching":
            self.fm_head = FMHead(config)
            self.direct_action_head = None
            self.action_expert = self.fm_head.action_expert
        elif config.action_head == "mlp":
            self.fm_head = None
            self.direct_action_head = MLPActionHead(config)
            self.action_expert = self.direct_action_head
        elif config.action_head == "query":
            self.fm_head = None
            self.direct_action_head = QueryActionHead(config)
            self.action_expert = self.direct_action_head
        else:
            raise ValueError(f"Unsupported action_head: {config.action_head}")
        self.future_latent_predictor = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self._queues: dict[str, deque[Tensor]] = {}
        self._ensemble_chunks: deque[tuple[Tensor, int]] = deque(maxlen=config.temporal_ensemble_max_chunks)
        self.reset()
        self.to(dtype=_torch_dtype(config.dtype))
        if config.device is not None:
            self.to(config.device)

    def reset(self) -> None:
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}
        self._ensemble_chunks = deque(maxlen=self.config.temporal_ensemble_max_chunks)

    def get_optim_params(self) -> Iterable[nn.Parameter]:
        return self.parameters()

    def load_compatible_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = (
            "visual_resampler.",
            "direct_action_head.",
            "metadata_modality",
            "subtask_modality",
            "metadata_encoder.",
            "future_latent_predictor.",
        )
        bad_missing = [
            key for key in missing if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        if bad_missing or unexpected:
            details = []
            if bad_missing:
                details.append(f"missing={bad_missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            raise RuntimeError("Checkpoint is not compatible with MiniVLAPolicy: " + ", ".join(details))

    def get_action_head(self) -> nn.Module:
        if self.config.action_head == "flow_matching":
            if self.fm_head is None:
                raise RuntimeError("MiniVLAConfig selects flow_matching but fm_head is not initialized")
            return self.fm_head
        if self.direct_action_head is None:
            raise RuntimeError(f"MiniVLAConfig selects {self.config.action_head} but direct_action_head is not initialized")
        return self.direct_action_head

    def sample_noise(self, shape: torch.Size | tuple[int, ...], device: torch.device) -> Tensor:
        if self.fm_head is not None:
            return self.fm_head.sample_noise(shape, device, next(self.parameters()).dtype)
        return torch.randn(shape, dtype=next(self.parameters()).dtype, device=device)

    def sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        if self.fm_head is not None:
            return self.fm_head.sample_time(batch_size, device)
        return torch.zeros(batch_size, dtype=torch.float32, device=device)

    def forward(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, Tensor | float]]:
        actions_raw = batch[ACTION]
        action_dim = batch.get(ACTION_DIM, self.config.action_dim)
        if torch.is_tensor(action_dim):
            if action_dim.numel() > 1 and not torch.all(action_dim == action_dim.flatten()[0]):
                raise ValueError("MiniVLA does not support mixed action_dim values within one batch")
            action_dim = int(action_dim.flatten()[0].item())
        action_dim = min(int(action_dim), self.config.max_action_dim)
        actions = pad_vector(actions_raw, self.config.max_action_dim).to(dtype=next(self.parameters()).dtype)
        obs_tokens = self.encode_observation_tokens(batch)
        loss, info = self.get_action_head().loss(
            obs_tokens=obs_tokens,
            actions=actions,
            action_dim=action_dim,
            noise=noise,
            time=time,
            action_pad_mask=get_action_pad_mask(batch),
            reduction=reduction,
        )
        future_loss = self.future_latent_loss(batch, obs_tokens)
        if future_loss is not None and reduction == "mean":
            loss = loss + self.config.future_latent_loss_weight * future_loss
            info["future_latent_loss"] = future_loss.detach()
            info["loss"] = loss.detach()
        return loss, info

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
        if self.config.use_temporal_ensemble:
            return self.select_action_temporal_ensemble(batch, noise=noise)
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    @torch.no_grad()
    def select_action_temporal_ensemble(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        new_chunk = self.predict_action_chunk(batch, noise=noise)
        self._ensemble_chunks.append((new_chunk, 0))

        candidates = []
        weights = []
        for chunk, age in self._ensemble_chunks:
            if age < chunk.shape[1]:
                candidates.append(chunk[:, age])
                weights.append(torch.exp(torch.tensor(-self.config.temporal_ensemble_decay * age, device=chunk.device)))
        if not candidates:
            raise RuntimeError("Temporal ensemble has no valid action candidates")

        stacked = torch.stack(candidates, dim=0)
        weight_tensor = torch.stack(weights).to(dtype=stacked.dtype)
        action = (stacked * weight_tensor[:, None, None]).sum(dim=0) / weight_tensor.sum().clamp_min(1e-6)
        self._ensemble_chunks = deque(
            [(chunk, age + 1) for chunk, age in self._ensemble_chunks if age + 1 < chunk.shape[1]],
            maxlen=self.config.temporal_ensemble_max_chunks,
        )
        return action

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        self.eval()
        obs_tokens = self.encode_observation_tokens(batch)
        actions = self.get_action_head().sample(obs_tokens=obs_tokens, noise=noise, num_steps=num_steps)
        return actions[..., : self.config.action_dim]

    def encode_context(self, batch: dict[str, Tensor]) -> Tensor:
        return self.encode_observation_tokens(batch).mean(dim=1, keepdim=True)

    def encode_observation_tokens(self, batch: dict[str, Tensor]) -> Tensor:
        image_tokens = self.encode_images(batch)
        if self.visual_resampler is not None:
            image_tokens = self.visual_resampler(image_tokens)
        elif self.config.max_image_tokens is not None and image_tokens.shape[1] > self.config.max_image_tokens:
            image_tokens = F.adaptive_avg_pool1d(
                image_tokens.transpose(1, 2),
                self.config.max_image_tokens,
            ).transpose(1, 2)
        image_tokens = image_tokens + self.image_modality

        text_tokens, text_mask = self.encode_text(batch)
        text_tokens = text_tokens + self.text_modality

        state_token = self.state_encoder(self.prepare_state(batch)) + self.state_modality
        optional_tokens, optional_mask = self.encode_optional_tokens(batch, state_token.shape[0], state_token.device)

        obs_tokens = torch.cat([image_tokens, text_tokens, state_token, optional_tokens], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros(
                    image_tokens.shape[:2],
                    dtype=torch.bool,
                    device=image_tokens.device,
                ),
                ~text_mask,
                torch.zeros(state_token.shape[:2], dtype=torch.bool, device=state_token.device),
                optional_mask,
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

    def encode_optional_tokens(
        self,
        batch: dict[str, Tensor],
        batch_size: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        tokens = []
        masks = []
        dtype = next(self.parameters()).dtype

        if self.config.use_episode_metadata and (EPISODE_SUCCESS in batch or EPISODE_QUALITY in batch):
            values = torch.zeros(batch_size, 2, dtype=dtype, device=device)
            if EPISODE_SUCCESS in batch:
                success = batch[EPISODE_SUCCESS]
                if not torch.is_tensor(success):
                    success = torch.as_tensor(success, device=device)
                values[:, 0] = success.to(device=device, dtype=dtype).flatten()[:batch_size]
            if EPISODE_QUALITY in batch:
                quality = batch[EPISODE_QUALITY]
                if not torch.is_tensor(quality):
                    quality = torch.as_tensor(quality, device=device)
                values[:, 1] = quality.to(device=device, dtype=dtype).flatten()[:batch_size]
            tokens.append(self.metadata_encoder(values)[:, None, :] + self.metadata_modality)
            masks.append(torch.zeros(batch_size, 1, dtype=torch.bool, device=device))

        if SUBTASK_TOKENS in batch:
            subtask_tokens = batch[SUBTASK_TOKENS].long()
            subtask_mask = batch.get(SUBTASK_ATTENTION_MASK)
            encoded, subtask_mask = self.text_encoder(subtask_tokens, subtask_mask)
            tokens.append(encoded + self.subtask_modality)
            masks.append(~subtask_mask)

        if not tokens:
            return (
                torch.zeros(batch_size, 0, self.config.hidden_dim, dtype=dtype, device=device),
                torch.zeros(batch_size, 0, dtype=torch.bool, device=device),
            )
        return torch.cat(tokens, dim=1), torch.cat(masks, dim=1)

    def future_latent_loss(self, batch: dict[str, Tensor], obs_tokens: Tensor) -> Tensor | None:
        if self.config.future_latent_loss_weight <= 0 or FUTURE_IMAGE not in batch:
            return None
        pred = self.future_latent_predictor(obs_tokens.mean(dim=1))
        with torch.no_grad():
            future_tokens = self.encode_images(batch, keys=(FUTURE_IMAGE,), add_camera=False)
            target = future_tokens.mean(dim=1)
        return F.mse_loss(pred, target)

    def encode_images(
        self,
        batch: dict[str, Tensor],
        keys: tuple[str, ...] | None = None,
        add_camera: bool = True,
    ) -> Tensor:
        image_tensors = self.prepare_images(batch, keys=keys)
        encoded = []
        for index, image in enumerate(image_tensors):
            if image.ndim == 5:
                batch_size, steps, channels, height, width = image.shape
                flat_image = image.reshape(batch_size * steps, channels, height, width)
                tokens = self.vision_encoder(flat_image)
                tokens = tokens.reshape(batch_size, steps, tokens.shape[1], tokens.shape[2])
                if add_camera:
                    camera_index = min(index, self.camera_embedding.num_embeddings - 1)
                    tokens = tokens + self.camera_embedding.weight[camera_index][None, None, None, :]
                temporal_count = min(steps, self.obs_temporal_embedding.num_embeddings)
                temporal_start = self.obs_temporal_embedding.num_embeddings - temporal_count
                temporal = self.obs_temporal_embedding.weight[temporal_start : temporal_start + temporal_count]
                if steps > temporal_count:
                    pad = temporal[:1].expand(steps - temporal_count, -1)
                    temporal = torch.cat([pad, temporal], dim=0)
                tokens = tokens + temporal.to(device=tokens.device, dtype=tokens.dtype)[None, :, None, :]
                encoded.append(tokens.flatten(1, 2))
            else:
                tokens = self.vision_encoder(image)
                if add_camera:
                    camera_index = min(index, self.camera_embedding.num_embeddings - 1)
                    tokens = tokens + self.camera_embedding.weight[camera_index][None, None, :]
                encoded.append(tokens)
        return torch.cat(encoded, dim=1)

    def prepare_images(self, batch: dict[str, Tensor], keys: tuple[str, ...] | None = None) -> list[Tensor]:
        if keys is None:
            keys = tuple(key for key in self.config.image_keys if key in batch)
        if not keys:
            keys = [key for key in batch if key == OBS_IMAGE or key.startswith(f"{OBS_IMAGES}.")]
        if not keys:
            raise KeyError(f"No image tensor found. Expected one of {self.config.image_keys}")

        images = []
        for key in keys:
            image = batch[key]
            if image.ndim not in {4, 5}:
                raise ValueError(f"Image {key} must have shape (B,C,H,W) or (B,T,C,H,W), got {tuple(image.shape)}")
            images.append(image.to(dtype=next(self.parameters()).dtype))
        return images

    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        if OBS_STATE not in batch:
            raise KeyError(f"Missing {OBS_STATE}")
        state = batch[OBS_STATE]
        if state.ndim == 3:
            state = state.flatten(start_dim=1)
        if state.ndim != 2:
            raise ValueError(f"{OBS_STATE} must have shape (B,D) or (B,T,D), got {tuple(state.shape)}")
        return pad_vector(state, self.config.max_state_dim).to(dtype=next(self.parameters()).dtype)
