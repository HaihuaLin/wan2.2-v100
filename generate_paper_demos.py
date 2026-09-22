"""
EvoSherlock 论文主题专属视频合成脚本
专为 Tesla V100-16GB 优化：
- 严格生成 5 秒（81 帧 @ 16 fps）
- 步数提升至 40 步，大幅提升画面细节与运动流畅度
- 内置 EvoSherlock 论文中 4 组经典安全事件与反事实干预案例
"""

import os
import argparse
import torch
from modelscope import snapshot_download

# 论文对应的 4 组经典安防与因果场景
CASES = {
    1: {
        "name": "theft_case (便利店盗窃行为 - 论文图6核心案例)",
        "prompt": "CCTV surveillance footage of a convenience store: a suspicious person in a dark jacket secretly slips merchandise from a shelf into their pocket while looking around cautiously, realistic security camera angle, sharp focus, 4k high quality",
        "output": "output/case1_theft.mp4"
    },
    2: {
        "name": "conflict_case (道路肢体冲突 - 论文图1/图6跨事件混淆案例)",
        "prompt": "Road surveillance video: two drivers stop on a city street and engage in an intense physical altercation and pushing beside a parked car, realistic traffic camera view, natural lighting, high dynamic range",
        "output": "output/case2_conflict.mp4"
    },
    3: {
        "name": "near_miss_case (机制干预 do(M) 近负样本：疑似盗窃但其实是捡拾归还)",
        "prompt": "High-angle street surveillance: a pedestrian accidentally drops a wallet on the sidewalk, another person behind quickly picks it up, taps their shoulder, and kindly returns it, clear natural motion, realistic CCTV quality",
        "output": "output/case3_near_miss_return.mp4"
    },
    4: {
        "name": "rainy_arson_case (环境干预 do(E)：夜雨低照度下的破坏/纵火隐患)",
        "prompt": "Nighttime street surveillance camera under heavy rain: streetlights reflecting on wet asphalt, a hooded figure attempting to force open a locked warehouse gate with tools, dramatic contrast, cinematic surveillance footage",
        "output": "output/case4_night_rain.mp4"
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate EvoSherlock-themed videos on V100")
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="./Wan2.2-TI2V-5B-Diffusers",
        help="本地模型路径或 ModelScope ID"
    )
    parser.add_argument(
        "--case", 
        type=int, 
        default=1, 
        choices=[1, 2, 3, 4],
        help="选择生成的论文案例 (1: 盗窃, 2: 街头冲突, 3: 归还物品假阳性, 4: 夜雨破坏)"
    )
    parser.add_argument(
        "--ultra", 
        action="store_true", 
        help="【画质天花板模式】自动配置: 720P (1280x720) 极清 + 50 步精细采样 + 3秒 (49帧)，不爆内存且画质最高"
    )
    parser.add_argument(
        "--num_frames", 
        type=int, 
        default=49, 
        choices=[49, 81],
        help="生成帧数: 49 (约3秒, 防容器内存溢出极稳档); 81 (约5秒满血档)"
    )
    parser.add_argument("--width", type=int, default=832, help="视频宽度 (建议: 832 或 1280)")
    parser.add_argument("--height", type=int, default=480, help="视频高度 (建议: 480 或 720)")
    parser.add_argument("--all", action="store_true", help="连续批量生成全部 4 个案例")
    parser.add_argument("--hd", action="store_true", help="开启 720P (1280x720) 模式")
    parser.add_argument("--steps", type=int, default=40, help="采样步数 (默认 40 步; --ultra 模式自动为 50 步)")
    return parser.parse_args()



def main():
    args = parse_args()
    os.makedirs("output", exist_ok=True)

    # 1. 确定模型路径
    if os.path.exists(args.model_id):
        model_path = args.model_id
    else:
        print(f"本地未找到 {args.model_id}，正在从魔搭验证/下载...")
        model_path = snapshot_download("Wan-AI/Wan2.2-TI2V-5B-Diffusers")

    # 2. 加载 WanPipeline
    print("=== 加载 Wan2.2 Pipeline 并注入 V100 16G 显存优化 ===")
    try:
        from diffusers import WanPipeline
        pipe_cls = WanPipeline
    except ImportError:
        from diffusers import AutoPipelineForText2Video
        pipe_cls = AutoPipelineForText2Video

    from diffusers.utils import export_to_video

    pipe = pipe_cls.from_pretrained(
        model_path,
        torch_dtype=torch.float16 # 严禁使用 bfloat16
    )

    # 开启针对 V100 16G 的显存与内存保护
    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    import gc

    # 3. 确定分辨率与时长参数
    if args.ultra:
        print("★ 已激活【画质天花板模式】: 720P (1280x720) + 50 步深度去噪 + 3.06秒 (49帧)")
        width = 1280
        height = 720
        steps = 50
        num_frames = 49
    else:
        num_frames = args.num_frames
        width = 1280 if args.hd else args.width
        height = 720 if args.hd else args.height
        steps = args.steps

    fps = 16
    duration = round(num_frames / fps, 2)
    selected_cases = [1, 2, 3, 4] if args.all else [args.case]

    for case_id in selected_cases:
        info = CASES[case_id]
        prompt = info["prompt"]
        if args.ultra:
            prompt += ", cinematic lighting, photorealistic, 8k uhd, sharp focus, masterpiece, crystal clear surveillance details"

        print(f"\n=======================================================")
        print(f"🎬 开始生成案例 {case_id}: {info['name']}")
        print(f"  时长: {duration} 秒 ({num_frames} 帧 @ {fps} fps)")
        print(f"  画质规格: {width}x{height} | 推理步数: {steps} 步")
        print(f"  提示词: {prompt}")
        print(f"=======================================================")

        gc.collect()
        torch.cuda.empty_cache()

        result = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=7.0
        )


        video_frames = result.frames[0]
        export_to_video(video_frames, info["output"], fps=fps)
        print(f"✓ 案例 {case_id} 视频已成功生成并导出至: {info['output']}")
        
        gc.collect()
        torch.cuda.empty_cache()


    print("\n🎉 全部指定视频生成任务完成！请在 output/ 目录下查看结果。")

if __name__ == "__main__":
    main()
