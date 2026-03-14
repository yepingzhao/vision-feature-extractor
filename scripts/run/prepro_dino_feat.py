#!/usr/bin/env python3

import sys
import os
import warnings
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings("ignore")

from pathlib import Path
from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm


# tag process_single_sample
def process_single_sample(args, processor, model, image_path):
    """处理单个样本数据"""
    # -------------------- 获取图像ID --------------------
    image_name = Path(image_path).stem
    image_id = image_name.split("_")[-1].lstrip("0")
    if image_id == "":
        image_id = "0"

    # -------------------- 检查是否已存在特征文件 --------------------
    npz_path = args.output / f"{image_id}.npz"
    if not args.force and npz_path.exists():
        np.load(npz_path)  # 检查文件是否能正常加载
        return

    # -------------------- 加载并处理图像 --------------------
    image_pil = Image.open(image_path).convert("RGB")

    # -------------------- 提取特征 --------------------
    device = model.device
    inputs = processor(images=image_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # -------------------- [1, 768] --------------------
    # image_features = outputs.last_hidden_state
    # features = image_features.cpu().numpy()

    # -------------------- [6, 257, 1024] --------------------
    features = []
    for feat in outputs.hidden_states:
        feat = feat.squeeze(0).cpu().numpy()
        features.append(feat)
    features = np.stack(features, axis=0)[1:]
    # features = np.mean(features, axis=0)

    # -------------------- 保存特征 --------------------
    np.savez(npz_path, feat=features)


# tag extract_visual_features_pipeline
def extract_visual_features_pipeline(args):
    """特征提取流程"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # -------------------- 获取所有图像文件路径 --------------------
    all = []
    for split in args.splits:
        all += list((Path(args.input) / split).rglob("*.jpg"))

    # -------------------- 加载模型 --------------------
    model = AutoModel.from_pretrained(args.model_id).to(device)
    model.eval()

    processor = AutoImageProcessor.from_pretrained(args.model_id, use_fast=True)

    # -------------------- 特征提取管道 --------------------
    pbar = tqdm(
        total=len(all), desc=f"preprocess {args.dataset}", ncols=80, colour="green"
    )

    for image_path in all:
        process_single_sample(args, processor, model, image_path)
        pbar.update(1)

    # -------------------- 清理模型以释放内存 --------------------
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pbar.close()


def main():
    parser = argparse.ArgumentParser(description="Extract DINOv2 features")
    parser.add_argument(
        "--model_id",  # option small, base, large, giant
        default="./models/dinov2-large",
    )
    parser.add_argument(
        "--dataset",
        default="ccmc",
    )
    parser.add_argument(
        "--input",
        default="./data/ccmc/images/",
    )
    parser.add_argument(
        "--splits",
        default=["ccmc_train", "ccmc_val", "ccmc_test"],
    )
    parser.add_argument(
        "--output",  # option
        default=Path("./data/ccmc/features/dinov2_large_l24"),
    )
    parser.add_argument("--force", action="store_true", default=True)  # option
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    extract_visual_features_pipeline(args)


# nohup python ./test/test_coco_clip.py > ./logs/coco_clip.log 2>&1 &
if __name__ == "__main__":
    main()
