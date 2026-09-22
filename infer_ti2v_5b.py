"""
Wan2.2-TI2V-5B 快速推理脚本
专为 Tesla V100-16GB 优化：
- 严格使用 torch.float16（避免 V100 无法运行 BF16）
- 启用 pipe.enable_model_cpu_offload() 预防显存溢出 (OOM)
- 启用 pipe.enable_vae_slicing() / tiling 减少显存峰值
"""

import os
import argparse
import torch
from modelscope import snapshot_download

def parse_args():
    parser = argparse.ArgumentParser(description="Wan2.2-TI2V-5B on V100 16GB")
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        help="ModelScope 上模型 ID 或本地路径"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="A cute cat wearing sunglasses driving a convertible car along the coast, cinematic, 4k",
        help="文生视频提示词"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        default=None,
        help="图生视频参考输入图片路径（可选）"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="output/result.mp4",
        help="视频保存路径"
    )
    parser.add_argument("--num_frames", type=int, default=49, help="生成帧数")
    parser.add_argument("--steps", type=int, default=30, help="推理步数")
    parser.add_argument("--fps", type=int, default=16, help="生成视频帧率")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else "output", exist_ok=True)

    print(f"=== 1. 正在获取模型: {args.model_id} ===")
    if os.path.exists(args.model_id):
        model_path = args.model_id
    else:
        print("正在从 ModelScope 下载/校验模型缓存...")
        model_path = snapshot_download(args.model_id)

    print("=== 2. 加载 Pipeline 并配置 V100 16G 显存优化 ===")
    from diffusers import AutoPipelineForText2Video, AutoPipelineForImage2Video
    from diffusers.utils import export_to_video
    from PIL import Image

    # 必须指定 float16（V100 缺乏 BF16 原生硬件单元）
    dtype = torch.float16

    if args.image:
        print(f"模式: 图生视频 (I2V)，参考图片: {args.image}")
        init_image = Image.open(args.image).convert("RGB")
        pipe = AutoPipelineForImage2Video.from_pretrained(
            model_path,
            torch_dtype=dtype
        )
    else:
        print("模式: 文生视频 (T2V)")
        pipe = AutoPipelineForText2Video.from_pretrained(
            model_path,
            torch_dtype=dtype
        )

    # 关键优化：16GB 显存核心防爆策略
    print("应用 CPU Offload 与 VAE Slicing 显存优化...")
    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    print(f"=== 3. 开始生成视频 (Prompt: {args.prompt}) ===")
    if args.image:
        result = pipe(
            image=init_image,
            prompt=args.prompt,
            num_frames=args.num_frames,
            num_inference_steps=args.steps
        )
    else:
        result = pipe(
            prompt=args.prompt,
            num_frames=args.num_frames,
            num_inference_steps=args.steps
        )

    video_frames = result.frames[0]
    export_to_video(video_frames, args.output, fps=args.fps)
    print(f"=== 4. 视频生成成功，已保存至: {args.output} ===")

if __name__ == "__main__":
    main()
