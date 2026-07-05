from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minivla.configuration_minivla import MiniVLAConfig


def masked_action_mse(
    pred_actions: Tensor,
    target_actions: Tensor,
    action_dim: int,
    action_pad_mask: Tensor | None = None,
    reduction: str = "mean",
) -> tuple[Tensor, Tensor]:
    per_dim_loss = F.mse_loss(
        pred_actions[..., :action_dim],
        target_actions[..., :action_dim],
        reduction="none",
    )
    if action_pad_mask is not None:
        if action_pad_mask.ndim == 1:
            action_pad_mask = action_pad_mask[:, None].expand(-1, target_actions.shape[1])
        if action_pad_mask.shape != target_actions.shape[:2]:
            raise ValueError(
                "action_pad_mask must have shape "
                f"{tuple(target_actions.shape[:2])}, got {tuple(action_pad_mask.shape)}"
            )
        valid = (~action_pad_mask.bool()).to(dtype=per_dim_loss.dtype, device=per_dim_loss.device)
        per_dim_loss = per_dim_loss * valid[:, :, None]
        denom = valid.sum().clamp_min(1.0) * action_dim
    else:
        denom = torch.tensor(per_dim_loss.numel(), dtype=per_dim_loss.dtype, device=per_dim_loss.device)

    if reduction == "mean":
        return per_dim_loss.sum() / denom, per_dim_loss
    if reduction == "none":
        return per_dim_loss, per_dim_loss
    raise ValueError(f"Unsupported reduction: {reduction}")


class MLPActionHead(nn.Module):
    """Direct BC baseline that maps pooled observation memory to an action chunk."""

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, int(config.hidden_dim * config.mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(config.hidden_dim * config.mlp_ratio), config.chunk_size * config.max_action_dim),
        )

    def sample(
        self,
        obs_tokens: Tensor,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        pooled = obs_tokens.mean(dim=1)
        actions = self.net(pooled)
        return actions.view(obs_tokens.shape[0], self.config.chunk_size, self.config.max_action_dim)

    def loss(
        self,
        obs_tokens: Tensor,
        actions: Tensor,
        action_dim: int,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        action_pad_mask: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, Tensor | float | str]]:
        actions = actions.to(dtype=obs_tokens.dtype, device=obs_tokens.device)
        pred_actions = self.sample(obs_tokens)
        loss, per_dim_loss = masked_action_mse(pred_actions, actions, action_dim, action_pad_mask, reduction)
        return loss, {
            "loss": loss.detach() if isinstance(loss, Tensor) else float(loss),
            "action_head": "mlp",
            "action_mse": per_dim_loss.detach().mean(),
        }


class QueryActionHead(nn.Module):
    """ACT-style query decoder baseline conditioned on observation memory."""

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.config = config
        self.query_embed = nn.Parameter(torch.zeros(1, config.chunk_size, config.hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.hidden_dim * config.mlp_ratio),
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_dit_layers)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.action_out = nn.Linear(config.hidden_dim, config.max_action_dim)

    def sample(
        self,
        obs_tokens: Tensor,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        queries = self.query_embed.expand(obs_tokens.shape[0], -1, -1).to(dtype=obs_tokens.dtype, device=obs_tokens.device)
        decoded = self.decoder(tgt=queries, memory=obs_tokens)
        return self.action_out(self.norm(decoded))

    def loss(
        self,
        obs_tokens: Tensor,
        actions: Tensor,
        action_dim: int,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        action_pad_mask: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, Tensor | float | str]]:
        actions = actions.to(dtype=obs_tokens.dtype, device=obs_tokens.device)
        pred_actions = self.sample(obs_tokens)
        loss, per_dim_loss = masked_action_mse(pred_actions, actions, action_dim, action_pad_mask, reduction)
        return loss, {
            "loss": loss.detach() if isinstance(loss, Tensor) else float(loss),
            "action_head": "query",
            "action_mse": per_dim_loss.detach().mean(),
        }
