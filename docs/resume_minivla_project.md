# MiniVLA Resume Wording

下面这版更贴合当前仓库主线：patch/CLIP image tokens、Observation Transformer、
DiT-style Flow Matching Action Expert、LeRobot 数据链路、policy server/client。

## 简历项目描述

**MiniVLA 自研与 SO-101 部署工程，LeRobot / PyTorch / Flow Matching**

- 面向 SO-101 六轴机械臂自研轻量 VLA policy，按 LeRobot 数据接口组织
  `observation.images.*`、`observation.state`、语言指令与 action chunk，打通数据预处理、
  训练、checkpoint、推理恢复和 HTTP policy server 链路。
- 参考 pi0/openpi 的 action-expert 解耦思路，实现多视角 image tokens、learnable visual token
  resampler、token-level language memory、state token 与可选 metadata/subtask tokens 的
  Observation Transformer 融合，并将 action head 配置化为 MLP baseline、ACT-style query
  decoder 与 DiT-style Flow Matching Head 三类可对比模块。
- 实现 masked action MSE、dataset stats normalizer、checkpoint 兼容加载、YAML 训练配置、
  `save_pretrained/from_pretrained`、延迟 benchmark、LeRobot 数据检查、离线评估与 SO-101
  client 安全框架，支持 action clamp、joint limit、EMA smoothing、temporal action ensemble、
  emergency stop 和推理频率统计。
- 围绕 VLA 部署稳定性整理 chunk_size、n_action_steps、denoise steps、视觉 backbone 冻结、
  动作平滑和数据质量 metadata 等关键变量，形成对轻量 VLA 从训练到真机控制闭环的工程理解。

## 面试时不要过度表述

- 不要说当前仓库已经复现完整 pi0；应说“参考 pi0/openpi 的工程形态与 FM action head 思路”。
- 不要说已经有大规模泛化结果；应说“当前重点是 SO-101 scale 的训练、部署和评估闭环”。
- 如果提到早期 CVAE / ActionQuery / ResNet18 版本，要说明那是历史迭代，不是当前 mainline。
