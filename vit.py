# vit.py

import torch
from vit_pytorch import ViT

def create_vit(dataset_name: str, num_classes: int):
    """
    根据数据集名称和类别数，创建一个配置好的ViT模型。
    """
    if dataset_name.lower() in ['cifar10', 'cifar100', 'gtsrb']:
        # 对于32x32的图像，使用较小的patch size
        print(f"Creating ViT model for {dataset_name.upper()} (32x32 input)...")
        model = ViT(
            image_size=32,
            patch_size=4,       # 32 / 4 = 8x8 patches
            num_classes=num_classes,
            dim=512,            # Dimension of tokens
            depth=6,            # Number of Transformer blocks
            heads=8,            # Number of attention heads
            mlp_dim=1024,       # Dimension of the MLP head
            dropout=0.1,
            emb_dropout=0.1
        )
    elif dataset_name.lower() == 'tinyimagenet':
        # 对于64x64的图像
        print(f"Creating ViT model for Tiny ImageNet (64x64 input)...")
        model = ViT(
            image_size=64,
            patch_size=8,       # 64 / 8 = 8x8 patches
            num_classes=num_classes,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1
        )
    else:
        raise ValueError(f"ViT configuration for dataset '{dataset_name}' is not defined.")

    return model