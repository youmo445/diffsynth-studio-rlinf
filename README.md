# Wan 世界模型微调项目 (DiffSynth-Studio)

这是基于 DiffSynth-Studio 开发的 Wan 视频生成模型微调代码，专门用于世界模型（World Model）训练推理。

## 主要内容

本项目包含了 Wan2.2-TI2V-5B 模型的训练与推理脚本，支持多帧条件输入和动作嵌入。

### 1. 训练脚本

运行以下脚本开始微调：

```bash
bash examples/wan_video/training/Wan2.2-TI2V-5B_rlinf.sh
```

### 2. 推理与评估

运行以下脚本进行batch_size=1的推理：

```bash
python examples/wanvideo/model_inference/Wan2.2-TI2V-5B-rlinf-bs_1.py
```

## 待更新

在训练时请将 diffsynth/models/model_manager.py 第118和第142行改为True

