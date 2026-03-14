#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import warnings
import argparse

warnings.filterwarnings("ignore")

import numpy as np
from tqdm import tqdm


def load_single_pre_fused_feature(args, feature_path):
    """加载单个预融合特征文件"""
    # -------------------- 获取图像ID --------------------
    image_id = Path(feature_path).stem

    # -------------------- 检查是否已存在输出特征文件 --------------------
    output_dir = args.output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{image_id}.npz"

    if not args.force and npz_path.exists():
        try:
            np.load(npz_path)  # 检查文件是否能正常加载
            return True
        except:
            pass  # 文件损坏，重新生成

    # -------------------- 加载预融合特征 --------------------
    try:
        data = np.load(feature_path)
        feat = data["feat"]

        # 直接使用预融合的特征
        save_dict = {}
        clip_mean = np.mean(feat[:24], axis=0)
        dino_mean = np.mean(feat[-6:], axis=0)
        feat = np.stack([clip_mean, dino_mean], axis=0)
        save_dict["feat"] = feat

        # -------------------- 保存特征 --------------------
        np.savez(npz_path, **save_dict)
        return True
    except Exception as e:
        print(f"Error processing {feature_path}: {e}")
        return False


def process_pre_fused_features(args):
    """处理预融合特征文件"""
    # -------------------- 获取所有预融合特征文件路径 --------------------
    all_features = list(Path(args.input_path).rglob("*.npz"))

    print(f"Found {len(all_features)} pre-fused feature files")

    # -------------------- 处理每个特征文件 --------------------
    pbar = tqdm(
        total=len(all_features),
        desc=f"Processing pre-fused features for {args.dataset}",
        ncols=80,
        colour="green",
    )

    success_count = 0
    for feature_path in all_features:
        if load_single_pre_fused_feature(args, feature_path):
            success_count += 1
        pbar.update(1)

    pbar.close()
    print(f"Successfully processed {success_count} feature files")


def main():
    parser = argparse.ArgumentParser(
        description="Load pre-fused visual features from .npz files"
    )
    parser.add_argument(
        "--dataset",
        default="ccmc",
    )
    parser.add_argument(
        "--input_path",
        default="./data/ccmc/features/mfm_att/",
        help="Path to pre-fused features directory (e.g., mfm_att)",
    )
    parser.add_argument(
        "--output_path",
        default="./data/ccmc/features/clip_dino_large_mean_stack",
        help="Path to output directory for loaded features",
    )
    parser.add_argument("--force", action="store_true", default=True)
    args = parser.parse_args()

    args.input_path = Path(args.input_path)
    args.output_path = Path(args.output_path)

    # 检查输入目录是否存在
    if not args.input_path.exists():
        print(f"Input directory does not exist: {args.input_path}")
        sys.exit(1)

    process_pre_fused_features(args)


if __name__ == "__main__":
    main()
