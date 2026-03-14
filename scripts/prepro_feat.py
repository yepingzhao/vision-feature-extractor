#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import warnings
import argparse
import json

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

dataset_dict = {
    "ccmc": {
        "input": "data/ccmc/images",
        "splits": ["ccmc_train", "ccmc_val", "ccmc_test"],
        "output": "data/ccmc/features",
    },
    "coco2014": {
        "input": "data/coco/coco2014",
        "splits": ["train2014", "val2014", "test2014"],
        "output": "data/coco/coco2014",
    },
}

model_dict = {
    "clip_vit_base_patch32": {
        "type": "clip",
        "size": "base",
        "path": "models/models--openai--clip-vit-base-patch32",
    },
    "clip_vit_large_patch14": {
        "type": "clip",
        "size": "large",
        "path": "models/models--openai--clip-vit-large-patch14",
    },
    "dinov2_small": {"type": "dinov2", "size": "small", "path": "models/dinov2-small"},
    "dinov2_base": {"type": "dinov2", "size": "base", "path": "models/dinov2-base"},
    "dinov2_large": {"type": "dinov2", "size": "large", "path": "models/dinov2-large"},
    "dinov2_giant": {"type": "dinov2", "size": "giant", "path": "models/dinov2-giant"},
    "mae_large": {"type": "mae", "size": "large", "path": "models/vit-mae-large"},
}


def process_single_sample(args, processor, model, image_path):
    """处理单个样本数据"""
    # -------------------- 获取图像ID --------------------
    image_name = Path(image_path).stem
    image_id = image_name.split("_")[-1].lstrip("0")
    if image_id == "":
        image_id = "0"

    # -------------------- 检查是否已存在特征文件 --------------------
    dataset_out = dataset_dict[args.dataset]["output"]
    model_type = model_dict[args.model]["type"]
    model_size = model_dict[args.model]["size"]

    mean = f"_mean" if args.mean else ""
    output_dir = f"{args.dataset}_{model_type}_{model_size}_l{args.levels}{mean}"
    output_path = Path(dataset_out) / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    npz_path = output_path / f"{image_id}.npz"

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
        if model_type == "clip":
            outputs = model.vision_model(**inputs, output_hidden_states=True)
        elif model_type == "dinov2":
            outputs = model(**inputs, output_hidden_states=True)
        elif model_type == "mae":
            outputs = model(**inputs, output_hidden_states=True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    features = []
    for feat in outputs.hidden_states:
        feat = feat.squeeze(0).cpu().numpy()
        features.append(feat)
    features = np.stack(features, axis=0)[-args.levels :]
    if args.mean:
        features = np.mean(features, axis=0)

    # -------------------- 保存特征 --------------------
    np.savez(npz_path, feat=features)


def load_model_and_processor(model_id):
    """加载所有配置的模型和处理器"""

    # 使用model_dict映射获取模型信息
    if model_id not in model_dict:
        raise ValueError(f"Unsupported model id: {model_id}")

    model_info = model_dict[model_id]
    model_type = model_info["type"]
    model_path = model_info["path"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_type} model '{model_id}' from {model_path}...")

    if model_type == "clip":
        from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel

        processor = CLIPProcessor.from_pretrained(model_path, use_fast=True)
        model = CLIPModel.from_pretrained(model_path).to(device)
    elif model_type == "dinov2":
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
        model = AutoModel.from_pretrained(model_path).to(device)
    elif model_type == "mae":
        from transformers import AutoImageProcessor, ViTMAEForPreTraining

        processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
        model = ViTMAEForPreTraining.from_pretrained(model_path).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.eval()
    return model, processor


def extract_visual_features_pipeline(args):
    """特征提取流程"""
    # -------------------- 获取所有图像文件路径 --------------------
    dataset_in = dataset_dict[args.dataset]["input"]
    splits = dataset_dict[args.dataset]["splits"]
    all_images = []
    for split in splits:
        all_images += list((Path(dataset_in) / split).rglob("*.jpg"))

    # -------------------- 加载所有模型 --------------------
    model, processor = load_model_and_processor(args.model)

    # -------------------- 特征提取管道 --------------------
    pbar = tqdm(
        total=len(all_images),
        desc=f"preprocess {args.dataset} with {args.model}",
        ncols=100,
        colour="green",
    )

    for image_path in all_images:
        process_single_sample(args, processor, model, image_path)
        pbar.update(1)

    pbar.close()

    # -------------------- 清理模型以释放内存 --------------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Extract visual features")
    parser.add_argument("--model", type=str, default="clip_vit_large_patch14") # clip_vit_large_patch14, dinov2_large, mae_large
    parser.add_argument("--levels", type=int, default=1) # 1,2,4,6,8,12,24
    parser.add_argument("--dataset", type=str, default="ccmc")
    parser.add_argument("--mean", type=int, default=0)
    parser.add_argument("--force", action="store_true", default=True) # False, True
    args = parser.parse_args()

    extract_visual_features_pipeline(args)


if __name__ == "__main__":
    main()

"""
du -sh data/ccmc/features/*

rm -rf data/ccmc/features/ccmc_clip_large_l4
rm -rf data/ccmc/features/ccmc_clip_large_l6
rm -rf data/ccmc/features/ccmc_clip_large_l8
rm -rf data/ccmc/features/ccmc_clip_large_l12
rm -rf data/ccmc/features/ccmc_clip_large_l24

# CLIP
CUDA_VISIBLE_DEVICES=0 nohup python scripts/prepro_feat.py --model clip_vit_large_patch14 --levels 1 --dataset ccmc > logs/ccmc_clip_large_l1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python scripts/prepro_feat.py --model clip_vit_large_patch14 --levels 2 --dataset ccmc > logs/ccmc_clip_large_l2.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python scripts/prepro_feat.py --model clip_vit_large_patch14 --levels 4 --dataset ccmc > logs/ccmc_clip_large_l4.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python scripts/prepro_feat.py --model clip_vit_large_patch14 --levels 6 --dataset ccmc > logs/ccmc_clip_large_l6.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python scripts/prepro_feat.py --model clip_vit_large_patch14 --levels 8 --dataset ccmc > logs/ccmc_clip_large_l8.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python scripts/prepro_feat.py --model clip_vit_large_patch14 --levels 12 --dataset ccmc > logs/ccmc_clip_large_l12.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python scripts/prepro_feat.py --model clip_vit_large_patch14 --levels 24 --dataset ccmc > logs/ccmc_clip_large_l24.log 2>&1 &

# mean
CUDA_VISIBLE_DEVICES=0 nohup python scripts/prepro_feat.py --model clip_vit_large_patch14 --levels 24 --mean 1 --dataset ccmc > logs/ccmc_clip_large_l24_mean.log 2>&1 &

# DINOv2
CUDA_VISIBLE_DEVICES=1 nohup python scripts/prepro_feat.py --model dinov2_large --levels 24 --dataset ccmc > logs/ccmc_dinov2_large_l24.log 2>&1 &

# MAE
CUDA_VISIBLE_DEVICES=1 nohup python scripts/prepro_feat.py --model mae_large --levels 24 --dataset ccmc > logs/ccmc_mae_large_l24.log 2>&1 &
"""
