# Wan2.2 V100 (16GB) 部署与快速推理工程

本项目专为 **阿里云 Tesla V100-SXM2-16GB**（Volta 架构）量身定制，用于稳定运行 **通义万相 Wan2.2-TI2V-5B**（图文生视频统一模型）。

---

## 针对 V100 16G 的核心优化策略

1. **显存防 OOM (Out Of Memory)**：
   - 启用 `pipe.enable_model_cpu_offload()`，将文本编码器与去噪网络动态调度至主机内存，大幅压低显存占用至 10GB~12GB。
   - 启用 `pipe.enable_vae_slicing()` 和 `pipe.enable_vae_tiling()`，避免解码高分辨率帧时显存瞬间冲顶。
2. **精度适配**：
   - 强制使用 `torch.float16`（规避 V100 不支持硬件级 `bfloat16` 导致的速度骤降或报错问题）。
3. **计算加速**：
   - 采用 `xFormers` 替代 V100 无法运行的 `FlashAttention-2`。

---

## 目录结构

```text
├── .gitignore          # 忽略大模型权重与生成视频
├── requirements.txt    # 依赖库列表
├── setup_v100.sh       # 阿里云服务器一键环境初始化脚本
├── infer_ti2v_5b.py    # 文生视频 / 图生视频推理脚本
└── README.md           # 使用说明
```

---

## 快速使用

### 1. 阿里云服务器一键安装环境
```bash
chmod +x setup_v100.sh
./setup_v100.sh
```

### 2. 文生视频 (Text-to-Video)
```bash
python infer_ti2v_5b.py \
  --prompt "A futuristic flying car cruising through neon-lit cyberpunk city, rainy night, cinematic lighting" \
  --output output/cyberpunk.mp4
```

### 3. 图生视频 (Image-to-Video)
```bash
python infer_ti2v_5b.py \
  --image input.jpg \
  --prompt "Camera slowly zooming in, glowing dynamic ambient light" \
  --output output/i2v_result.mp4
```
