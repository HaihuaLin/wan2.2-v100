#!/bin/bash
# Wan2.2 V100 (16GB) 一键环境适配脚本
# 专为阿里云 PAI-DSW 及 ECS V100 (CUDA 12.4 / Driver 550+) 优化

set -e

echo "=== 1. 检查基础环境 ==="
nvidia-smi

# 判断是否需要 sudo
if [ "$(id -u)" -eq 0 ]; then
    APT_CMD="apt-get"
else
    APT_CMD="sudo apt-get"
fi

echo "=== 2. 安装系统音视频基础库 ==="
$APT_CMD update && $APT_CMD install -y git git-lfs ffmpeg build-essential

# 配置 pip 使用阿里云国内镜像源加速
PIP_INDEX="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"

echo "=== 3. 检查并校准 PyTorch 与核心库 ==="
# 检测到 DSW 容器已自带 torch 2.5.0+cu124，无需重新下载几个G的 torch
python -c "import torch; print(f'当前检测到 PyTorch: {torch.__version__}, CUDA 可用: {torch.cuda.is_available()}')"

# 解决 DSW 自带 numpy 2.1.2 导致的二进制不兼容问题
echo "正在确保 numpy 兼容版本 (<2.0.0)..."
pip install "numpy>=1.24.0,<2.0.0" $PIP_INDEX

echo "=== 4. 安装 V100 显存优化加速库 (xFormers) ==="
# 安装与当前环境兼容的 xformers（若已存在或采用原生 SDPA 则作为加速增强）
pip install xformers $PIP_INDEX || echo "提示: 若 xformers 安装失败，PyTorch 2.5 原生内置的 SDPA 亦可完全胜任 V100 推理。"

echo "=== 5. 安装 Diffusers 及 Wan2.2 运行依赖 ==="
pip install -r requirements.txt $PIP_INDEX

echo "=== 6. 环境检测验证 ==="
python -c "import diffusers, transformers, modelscope; print('✓ 核心库验证通过！版本:', diffusers.__version__, transformers.__version__)"

echo "=== 全部就绪！现在可以运行推理测试 ==="
echo "文生视频示例："
echo "  python infer_ti2v_5b.py --prompt 'A cute cat playing in the snow' --width 832 --height 480"

