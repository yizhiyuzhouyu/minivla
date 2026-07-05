# MiniVLA

`minivla` 是一个独立的小型 VLA policy 包，接口尽量贴近 LeRobot batch 约定：

- image token: 多视角 CLIP/ViT patch tokens + linear projector + camera embedding
- text token: token embedding + attention mask，保留 token-level language memory
- state token: proprioception/state linear projection，使用 `observation.state`
- fusion: image/text/state/可选 metadata tokens concat 后接 observation Transformer
- FMHead: 参考 openpi/pi0 的 flow matching 训练和 Euler denoise 推理
- action expert: FMHead 内部使用轻量 DiT，以完整 observation memory conditioning 预测 action velocity

默认配置里保留了 `vlm_base_model_name="lerobot/smolvla_base"`，实际视觉编码器默认使用 `openai/clip-vit-base-patch32`。如果本机不能联网下载 HF 权重，可以把 `use_hf_vision_encoder=False`，会使用内置 patch-ViT fallback。

## Smoke Test

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/smoke_test.py
```

## Training

`scripts/train.py` 提供了一个 LeRobot 数据集训练入口，包含 dataset stats normalizer、checkpoint save/load 和基础命令行配置：

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/train.py \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --output-dir outputs/minivla \
  --batch-size 8 \
  --max-steps 10000
```

恢复训练：

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/train.py \
  --dataset-repo-id your_org/your_lerobot_dataset \
  --resume outputs/minivla/last.pt \
  --use-checkpoint-config
```

训练 checkpoint 会保存 `config`、`norm_stats`、兼容旧字段的 `dataset_stats`，以及 processor/tokenizer/normalizer metadata。推理时可以用同一套 transform 路径恢复：

```python
from minivla import MiniVLAPolicyRunner

runner = MiniVLAPolicyRunner.from_checkpoint("outputs/minivla/last.pt")
actions = runner.infer(observation)["actions"]
```

也可以启动轻量 HTTP policy server：

```bash
/home/yzyzy/miniconda3/envs/lerobot/bin/python scripts/serve_policy.py \
  --checkpoint outputs/minivla/last.pt \
  --port 8010
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

动作由 `FMHead` 生成：训练时采样噪声 `noise` 和时间 `t`，构造
`x_t = t * noise + (1 - t) * action`，让 DiT action expert 预测 velocity
`noise - action`；推理时从高斯噪声开始，用 Euler denoise 积分得到完整 action chunk。

loss 是 masked MSE，只计算有效 action step。batch 里可以传 LeRobot 风格的
`action_is_pad`，其中 `True` 表示该 step 是 padding，不参与 loss：

```python
batch["action_is_pad"] = torch.zeros(2, cfg.chunk_size, dtype=torch.bool)
batch["action_is_pad"][0, -2:] = True
```

`policy.fm_head` 可以单独使用：输入 observation memory tokens，输出完整 padded action chunk：

```python
obs_tokens = policy.encode_observation_tokens({k: v for k, v in batch.items() if k != ACTION})
padded_actions = policy.fm_head.sample(obs_tokens)
```

可选 metadata / auxiliary 输入：

- `episode.success`、`episode.quality` 会被编码成一个 episode metadata token。
- `subtask.label` 会被 processor tokenized 成 `subtask.tokens`，并作为额外 language memory。
- 设置 `future_latent_loss_weight > 0` 且 batch 提供 `future.image` 时，会启用轻量 future latent auxiliary loss。

## Notes

训练数据预处理可以直接沿用 LeRobot 的命名方式：图像键用 `observation.images.*`，关节/末端状态放 `observation.state`，动作放 `action`，语言 tokens 放 `observation.language.tokens`，mask 放 `observation.language.attention_mask`。`transforms.py` 负责 repack、resize、pad、normalize / unnormalize 和 tokenize 调度。
