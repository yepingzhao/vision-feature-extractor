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


def fuse_wrapper(fuse_type, features, model_type=None, stage=None):
    """
    根据不同阶段和参数返回不同的融合策略

    Args:
        fuse_type (str): 融合类型 ('mean', 'last', 'none')
        features (list): 特征列表
        model_type (str, optional): 模型类型 ('clip', 'dino')
        stage (str, optional): 融合阶段 ('intra' for 单模型内融合, 'inter' for 多模型间融合)

    Returns:
        numpy.ndarray: 融合后的特征
    """
    if not isinstance(features, np.ndarray):
        features = np.stack(features, axis=0)

    if fuse_type == "mean":
        if stage == "intra":
            if model_type == "clip":
                # 对CLIP模型，忽略第一层特征，对剩余特征取平均
                return np.mean(features[1:], axis=0)
            elif model_type == "dino":
                # 对DINO模型，对最后6层特征取平均
                return np.mean(features[-6:], axis=0)
        elif stage == "inter":
            # 对多模型间特征取平均
            return np.mean(features, axis=0)
        else:
            # 默认情况，对所有特征取平均
            return np.mean(features, axis=0)

    elif fuse_type == "last":
        # 返回最后一层特征
        return features[-1]

    elif fuse_type == "none":
        # 不进行融合，返回所有特征
        return features

    else:
        raise ValueError(f"Unsupported fuse type: {fuse_type}")


def process_single_sample(args, processors, models, image_path):
    """处理单个样本数据"""
    # -------------------- 获取图像ID --------------------
    image_name = Path(image_path).stem
    image_id = image_name.split("_")[-1].lstrip("0")
    if image_id == "":
        image_id = "0"

    # -------------------- 检查是否已存在特征文件 --------------------
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{image_id}.npz"

    if not args.force and npz_path.exists():
        np.load(npz_path)  # 检查文件是否能正常加载
        return

    # -------------------- 加载并处理图像 --------------------
    image_pil = Image.open(image_path).convert("RGB")

    all_features = {}

    # -------------------- 提取所有模型的特征 --------------------
    for key in models.keys():
        model = models[key]
        processor = processors[key]
        model_info = model_dict[key]
        model_type = model_info["type"]

        device = model.device
        inputs = processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if model_type == "clip":
                features = []
                outputs = model.vision_model(**inputs, output_hidden_states=True)
                for feat in outputs.hidden_states:
                    feat = feat.squeeze(0).cpu().numpy()
                    features.append(feat)
                # 使用 fuse_wrapper 控制单模型内多层次特征融合策略
                processed_features = fuse_wrapper(
                    args.fuse[0], features, model_type="clip", stage="intra"
                )
                all_features[key] = processed_features
            elif model_type == "dino":
                features = []
                outputs = model(**inputs, output_hidden_states=True)
                for feat in outputs.hidden_states:
                    feat = feat.squeeze(0).cpu().numpy()
                    features.append(feat)
                # 使用 fuse_wrapper 控制单模型内多层次特征融合策略
                processed_features = fuse_wrapper(
                    args.fuse[0], features, model_type="dino", stage="intra"
                )
                all_features[key] = processed_features
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

    # -------------------- 特征融合逻辑 --------------------
    save_dict = {}

    # 使用 fuse_wrapper 控制多模型间特征融合策略
    features_list = list(all_features.values())

    if args.fuse[1] == "concat":
        # 简单拼接融合
        fused_features = np.concatenate(features_list, axis=-1)
        save_dict["feat"] = fused_features
    elif args.fuse[1] == "mean":
        # 平均融合（需要确保特征维度一致）
        fused_features = fuse_wrapper(args.fuse[1], features_list, stage="inter")
        save_dict["feat"] = fused_features
    elif args.fuse[1] == "stack":
        # 堆叠融合
        fused_features = np.stack(features_list, axis=0)
        save_dict["feat"] = fused_features[0][-6:]
    elif args.fuse[1] == "none":
        # 不融合，只保存各模型特征
        save_dict = all_features

    # -------------------- 保存特征 --------------------
    np.savez(npz_path, **save_dict)


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

    # -------------------- 为每个图像执行特征提取 --------------------
    pbar = tqdm(
        total=len(all_images),
        desc=f"preprocess fused features for {args.dataset}",
        ncols=80,
        colour="green",
    )

    for image_path in all_images:
        process_single_sample(args, processors, models, image_path)
        pbar.update(1)

    pbar.close()

    # -------------------- 清理模型以释放内存 --------------------
    # for key in models.keys():
    #     del models[key]
    #     del processors[key]
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
        default=[
            # "clip_vit_large_patch14",
            "dinov2_large",
        ],  # clip_vit_large_patch14, dinov2_large
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
        # default="./data/ccmc/features/clip_large_l24",
        default="./data/ccmc/features/dinov2_large_l6",
    )
    parser.add_argument(
        "--fuse",
        nargs="+",
        default=["none", "stack"],
        help="Feature fusion strategy: fuse[0] controls intra-model fusion (mean, last, none), fuse[1] controls inter-model fusion (concat, mean, stack, none)",
    )
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()

    args.output = Path(f"{args.output}")

    extract_visual_features_pipeline(args)


if __name__ == "__main__":
    main()
