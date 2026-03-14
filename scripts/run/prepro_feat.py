#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import warnings
import argparse
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
from tqdm import tqdm

# 定义模型ID到模型路径和类型的映射
model_dict = {
    "clip_vit_base_patch32": {
        "type": "clip",
        "path": "./models/models--openai--clip-vit-base-patch32",
    },
    "clip_vit_large_patch14": {
        "type": "clip",
        "path": "./models/models--openai--clip-vit-large-patch14",
    },
    "dinov2_small": {"type": "dino", "path": "./models/dinov2-small"},
    "dinov2_base": {"type": "dino", "path": "./models/dinov2-base"},
    "dinov2_large": {"type": "dino", "path": "./models/dinov2-large"},
    "dinov2_giant": {"type": "dino", "path": "./models/dinov2-giant"},
}


def process_single_sample(args, processor, model, image_path, model_key):
    """处理单个样本数据"""
    # -------------------- 获取图像ID --------------------
    image_name = Path(image_path).stem
    image_id = image_name.split("_")[-1].lstrip("0")
    if image_id == "":
        image_id = "0"

    # -------------------- 检查是否已存在特征文件 --------------------
    # output_dir = args.output / model_key
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{image_id}.npz"

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
        if args.model_type == "clip":
            outputs = model.vision_model(**inputs, output_hidden_states=True)
        elif args.model_type == "dino":
            outputs = model(**inputs, output_hidden_states=True)
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")

    features = []
    for feat in outputs.hidden_states:
        feat = feat.squeeze(0).cpu().numpy()
        features.append(feat)
    features = np.stack(features, axis=0)[1:]

    # -------------------- 保存特征 --------------------
    np.savez(npz_path, feat=features)


def load_models_and_processors(model_ids):
    """加载所有配置的模型和处理器"""
    models = {}
    processors = {}

    for model_id in model_ids:
        # 使用model_dict映射获取模型信息
        if model_id not in model_dict:
            raise ValueError(f"Unsupported model id: {model_id}")

        model_info = model_dict[model_id]
        model_type = model_info["type"]
        model_path = model_info["path"]
        key = model_id  # 使用model_id作为key

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading {model_type} model '{model_id}' from {model_path}...")

        if model_type == "clip":
            models[key] = CLIPModel.from_pretrained(model_path).to(device)
            processors[key] = CLIPProcessor.from_pretrained(model_path, use_fast=True)
        elif model_type == "dino":
            models[key] = AutoModel.from_pretrained(model_path).to(device)
            processors[key] = AutoImageProcessor.from_pretrained(
                model_path, use_fast=True
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        models[key].eval()

    return models, processors


def extract_visual_features_pipeline(args):
    """特征提取流程"""
    # -------------------- 获取所有图像文件路径 --------------------
    all_images = []
    for split in args.splits:
        all_images += list((Path(args.input) / split).rglob("*.jpg"))

    # -------------------- 加载所有模型 --------------------
    models, processors = load_models_and_processors(args.models)

    # -------------------- 为每个模型执行特征提取 --------------------
    for key in models.keys():
        # 从model_dict获取模型类型
        model_info = model_dict[key]
        args.model_type = model_info["type"]  # 临时设置用于process_single_sample

        # -------------------- 特征提取管道 --------------------
        pbar = tqdm(
            total=len(all_images),
            desc=f"preprocess {key} for {args.dataset}",
            ncols=80,
            colour="green",
        )

        for image_path in all_images:
            process_single_sample(args, processors[key], models[key], image_path, key)
            pbar.update(1)

        pbar.close()

    # -------------------- 清理模型以释放内存 --------------------
    for key in models.keys():
        del models[key]
        del processors[key]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Extract visual features with multiple models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        # required=True,
        default=["clip_vit_large_patch14", "dinov2_large"],
        help="Model IDs to use (e.g. clip_vit_large_patch14 dinov2_large)",
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
        "--output",
        default=Path("./data/ccmc/features/ccmc_clip_large_l2"),
    )
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    extract_visual_features_pipeline(args)


if __name__ == "__main__":
    main()
