"""
Wan2.2-TI2V-5B 快速推理脚本
专为 Tesla V100-16GB 优化：
- 严格使用 torch.float16（避免 V100 无法运行 BF16）
- 启用 pipe.enable_model_cpu_offload() 预防显存溢出 (OOM)
- 启用 pipe.enable_vae_slicing() / tiling 减少显存峰值
- 默认采用 832*480（480P），提供最高稳定性；支持手动指定 1280*720
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
        help="视频提示词"
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
    parser.add_argument("--width", type=int, default=832, help="视频宽度 (建议: 832 或 1280)")
    parser.add_argument("--height", type=int, default=480, help="视频高度 (建议: 480 或 720)")
    parser.add_argument("--num_frames", type=int, default=49, help="生成帧数 (通常为 49 或 81)")
    parser.add_argument("--steps", type=int, default=30, help="推理步数 (默认 30)")
    parser.add_argument("--fps", type=int, default=16, help="生成视频帧率")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else "output", exist_ok=True)

    print(f"=== 1. 正在获取/验证模型: {args.model_id} ===")
    if os.path.exists(args.model_id):
        model_path = args.model_id
    else:
        print("正在从 ModelScope 校验或下载模型缓存...")
        model_path = snapshot_download(args.model_id)

    print(f"模型本地加载路径: {model_path}")

    print("=== 2. 加载 Pipeline 并配置 V100 16G 显存优化 ===")
    # 动态适配 Wan 专用 Pipeline 或通用 Pipeline
    try:
        from diffusers import WanPipeline, WanImageToVideoPipeline
        text_pipe_cls = WanPipeline
        img_pipe_cls = WanImageToVideoPipeline
    except ImportError:
        from diffusers import AutoPipelineForText2Video, AutoPipelineForImage2Video
        text_pipe_cls = AutoPipelineForText2Video
        img_pipe_cls = AutoPipelineForImage2Video

    from diffusers.utils import export_to_video
    from PIL import Image

    # 关键：V100 必须指定 float16（缺少硬件 BF16 单元）
    dtype = torch.float16

    if args.image:
        print(f"模式: 图生视频 (I2V)，参考图片: {args.image}")
        init_image = Image.open(args.image).convert("RGB")
        pipe = img_pipe_cls.from_pretrained(
            model_path,
            torch_dtype=dtype
        )
    else:
        print("模式: 文生视频 (T2V)")
        pipe = text_pipe_cls.from_pretrained(
            model_path,
            torch_dtype=dtype
        )

    # 显存防 OOM 核心配置
    print("应用 CPU Offload 与 VAE Slicing/Tiling 显存保护...")
    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    print(f"=== 3. 开始生成视频 ===")
    print(f"  分辨率: {args.width}x{args.height}")
    print(f"  帧数: {args.num_frames} 帧, 步数: {args.steps}")
    print(f"  提示词: {args.prompt}")

    gen_kwargs = {
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
        "num_inference_steps": args.steps,
    }
    if args.image:
        gen_kwargs["image"] = init_image

    result = pipe(**gen_kwargs)
    video_frames = result.frames[0]

    export_to_video(video_frames, args.output, fps=args.fps)
    print(f"=== 4. 视频生成成功！已保存至: {args.output} ===")

if __name__ == "__main__":
    main()

