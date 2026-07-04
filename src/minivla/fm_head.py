from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from minivla.configuration_minivla import MiniVLAConfig


def sample_beta(alpha: float, beta: float, batch_size: int, device: torch.device) -> Tensor:
    dist = torch.distributions.Beta(torch.tensor(alpha), torch.tensor(beta))
    return dist.sample((batch_size,)).to(device=device, dtype=torch.float32)


def create_sinusoidal_pos_embedding(
    time: Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device,
) -> Tensor:
    if dimension % 2 != 0:
        raise ValueError("dimension must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time must have shape (batch_size,)")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float64, device=device)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid = (2 * math.pi / period)[None, :] * time[:, None].to(torch.float64)
    return torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=1).to(torch.float32)


class DiTActionExpert(nn.Module):
    """DiT-style velocity network used inside the flow-matching head."""

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.config = config
        self.action_in_proj = nn.Linear(config.max_action_dim, config.hidden_dim)
        self.action_out_proj = nn.Linear(config.hidden_dim, config.max_action_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.action_pos_embed = nn.Parameter(torch.zeros(1, config.chunk_size, config.hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.hidden_dim * config.mlp_ratio),
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_dit_layers)
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, context_token: Tensor, noisy_actions: Tensor, time: Tensor) -> Tensor:
        time_emb = create_sinusoidal_pos_embedding(
            time,
            self.config.hidden_dim,
            self.config.min_period,
            self.config.max_period,
            noisy_actions.device,
        ).to(dtype=noisy_actions.dtype)
        time_emb = self.time_mlp(time_emb)[:, None, :]

        action_tokens = self.action_in_proj(noisy_actions)
        action_tokens = action_tokens + time_emb + self.action_pos_embed[:, : action_tokens.shape[1]]
        tokens = torch.cat([context_token, action_tokens], dim=1)
        tokens = self.norm(self.transformer(tokens))
        return self.action_out_proj(tokens[:, -self.config.chunk_size :])


class FMHead(nn.Module):
    """Flow-matching action head.

    It trains the action expert to predict the velocity field
    ``u_t = noise - action`` at interpolated samples
    ``x_t = t * noise + (1 - t) * action``. During inference it starts
    from Gaussian noise at t=1 and integrates backward to t=0.
    """

    def __init__(self, config: MiniVLAConfig):
        super().__init__()
        self.config = config
        self.action_expert = DiTActionExpert(config)

    def sample_noise(self, shape: torch.Size | tuple[int, ...], device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.randn(shape, dtype=dtype, device=device)

    def sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        time = sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            batch_size,
            device,
        )
        return time * self.config.time_sampling_scale + self.config.time_sampling_offset

    def denoise_step(self, context: Tensor, x_t: Tensor, time: Tensor) -> Tensor:
        return self.action_expert(context, x_t, time)

    def loss(
        self,
        context: Tensor,
        actions: Tensor,
        action_dim: int,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        action_pad_mask: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, Tensor | float]]:
        if actions.shape[1] != self.config.chunk_size:
            raise ValueError(f"Expected action chunk length {self.config.chunk_size}, got {actions.shape[1]}")

        actions = actions.to(dtype=context.dtype, device=context.device)
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device, actions.dtype)
        noise = noise.to(device=actions.device, dtype=actions.dtype)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        x_t = time[:, None, None].to(actions.dtype) * noise + (1.0 - time[:, None, None].to(actions.dtype)) * actions
        target_velocity = noise - actions
        pred_velocity = self.denoise_step(context, x_t, time)

        per_dim_loss = F.mse_loss(
            pred_velocity[..., :action_dim],
            target_velocity[..., :action_dim],
            reduction="none",
        )
        if action_pad_mask is not None:
            valid = (~action_pad_mask.bool()).to(dtype=per_dim_loss.dtype, device=per_dim_loss.device)
            per_dim_loss = per_dim_loss * valid[:, :, None]
            denom = valid.sum().clamp_min(1.0) * action_dim
        else:
            denom = torch.tensor(per_dim_loss.numel(), dtype=per_dim_loss.dtype, device=per_dim_loss.device)

        if reduction == "mean":
            loss = per_dim_loss.sum() / denom
        elif reduction == "none":
            loss = per_dim_loss
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

        info = {
            "loss": loss.detach() if isinstance(loss, Tensor) else float(loss),
            "fm_time_mean": time.detach().mean(),
            "fm_noise_std": noise.detach().float().std(),
        }
        return loss, info

    @torch.no_grad()
    def sample(
        self,
        context: Tensor,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")

        batch_size = context.shape[0]
        device = context.device
        dtype = context.dtype
        if noise is None:
            noise = self.sample_noise(
                (batch_size, self.config.chunk_size, self.config.max_action_dim),
                device,
                dtype,
            )
        x_t = noise.to(device=device, dtype=dtype)
        dt = -1.0 / num_steps

        for step in range(num_steps):
            time = torch.full((batch_size,), 1.0 + step * dt, dtype=torch.float32, device=device)
            velocity = self.denoise_step(context, x_t, time)
            x_t = x_t + dt * velocity
        return x_t

