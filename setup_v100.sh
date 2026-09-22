#!/bin/bash
# Wan2.2 V100 (16GB) 一键环境配置脚本
# 适用于阿里云 ECS V100 实例 (CUDA 12.4 / Driver 550+)

set -e

echo "=== 1. 检查基础环境 ==="
nvidia-smi

echo "=== 2. 安装系统基础库 ==="
sudo apt-get update && sudo apt-get install -y git git-lfs ffmpeg build-essential

echo "=== 3. 安装 PyTorch (针对 CUDA 12.4) 与 xFormers ==="
# 注意：V100 不支持 FlashAttention-2，必须使用 xFormers 作为加速后端
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install xformers --index-url https://download.pytorch.org/whl/cu124

echo "=== 4. 安装 Diffusers 及模型运行依赖 ==="
pip install -r requirements.txt

echo "=== 5. 环境配置完成！==="
echo "现在可以使用以下命令进行推理："
echo "  python infer_ti2v_5b.py --prompt 'A cute cat playing in the snow'"
