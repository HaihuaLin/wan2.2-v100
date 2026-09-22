"""
Wan2.2-TI2V-5B-Diffusers 模型提前下载脚本 (ModelScope 魔搭)
阿里云内网极速拉取，带实时下载进度条
"""

import sys
from modelscope import snapshot_download

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
LOCAL_DIR = "./Wan2.2-TI2V-5B-Diffusers"

def main():
    print(f"=== 开始从魔搭 (ModelScope) 下载模型: {MODEL_ID} ===")
    print(f"本地保存目录: {LOCAL_DIR}")
    print("在阿里云 PAI-DSW 环境下通常享有内网/专线高速 (50MB/s ~ 100MB/s+)...")
    
    try:
        path = snapshot_download(
            model_id=MODEL_ID,
            local_dir=LOCAL_DIR
        )
        print("\n==============================================")
        print(f"✓ 模型下载完成！路径: {path}")
        print("现在可以直接运行推理：")
        print(f"  python infer_ti2v_5b.py --model_id {LOCAL_DIR} --prompt 'A cute cat in snowy park'")
        print("==============================================")
    except Exception as e:
        print(f"下载过程中出错: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
