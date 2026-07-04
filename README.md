# MiniVLA

`minivla` 是一个独立的小型 VLA policy 包，接口尽量贴近 LeRobot batch 约定：

- image token: CLIP/ViT vision encoder + linear projector
- text token: token embedding + attention-mask mean pooling + linear
- state token: proprioception/state linear projection，使用 `observation.state`
- fusion: pooled image/text/state token concat 后接 MLP
- FMHead: 参考 openpi/pi0 的 flow matching 训练和 Euler denoise 推理
- action expert: FMHead 内部使用轻量 DiT 预测 action velocity

默认配置里保留了 `vlm_base_model_name="lerobot/smolvla_base"`，实际视觉编码器默认使用 `openai/clip-vit-base-patch32`。如果本机不能联网下载 HF 权重，可以把 `use_hf_vision_encoder=False`，会使用内置 patch-ViT fallback。

## Smoke Test

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/smoke_test.py
```

## Minimal Usage

```python
import torch
from minivla import MiniVLAConfig, MiniVLAPolicy
from minivla.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

cfg = MiniVLAConfig(
    use_hf_vision_encoder=False,
    image_keys=("observation.images.front",),
    max_state_dim=14,
    max_action_dim=14,
    action_dim=14,
    chunk_size=16,
)
policy = MiniVLAPolicy(cfg)

batch = {
    "observation.images.front": torch.rand(2, 3, 224, 224),
    OBS_LANGUAGE_TOKENS: torch.randint(0, cfg.text_vocab_size, (2, 12)),
    OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 12, dtype=torch.bool),
    OBS_STATE: torch.randn(2, 14),
    ACTION: torch.randn(2, cfg.chunk_size, 14),
}

loss, info = policy(batch)
actions = policy.predict_action_chunk({k: v for k, v in batch.items() if k != ACTION})
```

`policy.fm_head` 可以单独使用：输入已经融合好的 context token，输出完整 padded action chunk：

```python
context = policy.encode_context({k: v for k, v in batch.items() if k != ACTION})
padded_actions = policy.fm_head.sample(context)
```

## Notes

训练数据预处理可以直接沿用 LeRobot 的命名方式：图像键用 `observation.images.*`，关节/末端状态放 `observation.state`，动作放 `action`，语言 tokens 放 `observation.language.tokens`，mask 放 `observation.language.attention_mask`。
