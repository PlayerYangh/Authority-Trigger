"""
Evaluate Robustness Against a Pre-Computed Adaptive Trigger

This script is designed to test the effectiveness of a hardened/robust model
against a *previously generated* adaptive attack trigger.

It loads:
1.  A robust model (e.g., one trained with randomized smoothing).
2.  A trigger (mask + pattern) saved by 'adaptive_attack.py'.
3.  A clean test dataset.

It then applies the saved trigger to the clean test images and feeds them 
into the robust model, reporting the final accuracy. A high accuracy indicates
the trigger *still works* (defense failed), while a low accuracy 
(near random guess) indicates the defense was *successful*.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
from torchvision.datasets.folder import IMG_EXTENSIONS

import sys
import os

current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
root_dir = os.path.dirname(current_dir)

if root_dir not in sys.path:
    sys.path.append(root_dir)
from Models.resnet import ResNet18, ResNet50
from Models.vgg import VGG16_CIFAR100
from Models.vit import create_vit
from Common.utils import Accuracy

class UnNormalize(object):
    """
    Reverses the T.Normalize transformation.
    This works by applying the inverse operation of normalization.
    """
    def __init__(self, mean, std):
        # Inverse of Normalize(mean, std) is
        # UnNormalize = Normalize(mean = -mean/std, std = 1/std)
        self.unnormalize = T.Normalize(
            mean=[-m/s for m, s in zip(mean, std)],
            std=[1/s for s in std]
        )
    def __call__(self, tensor):
        return self.unnormalize(tensor)

def get_dataset_params(dataset_name):
    """Returns normalization stats, class count, and crop size for a given dataset."""
    if dataset_name == 'cifar10':
        mean, std, num_classes, crop_size = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 10, 32
    elif dataset_name == 'cifar100':
        mean, std, num_classes, crop_size = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 100, 32
    elif dataset_name == 'gtsrb':
        mean, std, num_classes, crop_size = (0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669), 43, 32
    elif dataset_name == 'tinyimagenet':
        # Note: Crop size is 32 to match the main training script's default resize
        mean, std, num_classes, crop_size = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 200, 32
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return mean, std, num_classes, crop_size

def evaluate_with_saved_trigger(args):
    """
    Loads a robust model and evaluates its accuracy when attacked with a
    pre-computed adaptive trigger.
    """
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    global IMG_EXTENSIONS
    if '.JPEG' not in IMG_EXTENSIONS:
        IMG_EXTENSIONS = IMG_EXTENSIONS + ('.JPEG',)

    mean, std, num_classes, crop_size = get_dataset_params(args.dataset)

    # 1. Load the robust (hardened) model
    if args.model == 'resnet18': net = ResNet18(num_classes=num_classes)
    elif args.model == 'resnet50': net = ResNet50(num_classes=num_classes)
    elif args.model == 'vgg16': net = VGG16_CIFAR100(num_classes=num_classes)
    elif args.model == 'vit': net = create_vit(args.dataset, num_classes)
    else: raise ValueError(f"Unsupported model architecture: {args.model}")

    if not os.path.exists(args.robust_model_path):
        raise FileNotFoundError(f"Robust model file not found at {args.robust_model_path}")
    print(f"Loading robust model from: {args.robust_model_path}")
    net.load_state_dict(torch.load(args.robust_model_path, map_location=device, weights_only=True))
    net.eval()
    net.to(device)

    # 2. Load the pre-computed adaptive mask and pattern
    print(f"Loading adaptive trigger from: {args.trigger_dir}")
    try:
        # Mask is saved as a single-channel grayscale image
        mask_path = os.path.join(args.trigger_dir, 'adaptive_mask.png')
        mask_pil = Image.open(mask_path).convert('L')
        # Convert mask to a [1, H, W] tensor in [0, 1] range
        mask = T.ToTensor()(mask_pil).to(device)
        
        # Pattern is saved as a 3-channel RGB image
        pattern_path = os.path.join(args.trigger_dir, 'adaptive_pattern.png')
        pattern_pil = Image.open(pattern_path)
        # Convert pattern to a [C, H, W] tensor in [0, 1] range
        pattern = T.ToTensor()(pattern_pil).to(device)
        
        print("Adaptive trigger (mask and pattern) loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Could not find trigger files in '{args.trigger_dir}'.")
        print(f"Please ensure 'adaptive_mask.png' and 'adaptive_pattern.png' exist. Details: {e}")
        return

    # 3. Load the clean test dataset
    transform_test = T.Compose([T.Resize((crop_size, crop_size)), T.ToTensor(), T.Normalize(mean, std)])
    clean_dataset = ImageFolder(root=args.clean_data_dir, transform=transform_test)
    clean_loader = DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Loaded {len(clean_dataset)} clean images from '{args.clean_data_dir}' for evaluation.")

    # 4. Apply the saved trigger and evaluate
    unnormalize = UnNormalize(mean, std)
    normalize = T.Normalize(mean, std)
    
    attack_accuracy = Accuracy()
    
    print("\nApplying saved adaptive trigger and evaluating...")
    with torch.no_grad():
        for images_norm, true_labels in tqdm(clean_loader, desc="Evaluating"):
            images_norm, true_labels = images_norm.to(device), true_labels.to(device)
            
            # Un-normalize images to pixel space [0, 1]
            images_unnorm = unnormalize(images_norm)

            # Apply trigger: (1-M) * X + M * P
            # We use the mask shape [1, H, W] which broadcasts over the batch and channels
            triggered_images_unnorm = (1 - mask) * images_unnorm + mask * pattern
            triggered_images_unnorm = torch.clamp(triggered_images_unnorm, 0.0, 1.0)
            
            # Re-normalize for model input
            triggered_images_norm = normalize(triggered_images_unnorm)
            
            output = net(triggered_images_norm)
            attack_accuracy.update(output, true_labels)

    print("\n--- Evaluation Finished ---")
    print(f"Model evaluated: {args.robust_model_path}")
    print(f"Trigger used: from '{args.trigger_dir}'")
    print(f"Accuracy after applying the saved adaptive trigger: {attack_accuracy}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a robust model against a pre-computed adaptive attack trigger.")
    
    # --- Path Arguments ---
    parser.add_argument('--robust_model_path', type=str, required=True, 
                        help='Path to the ROBUST benign backdoor model file (e.g., trained with noise).')
    parser.add_argument('--clean_data_dir', type=str, required=True, 
                        help="Path to the clean test dataset directory (ImageFolder format).")
    parser.add_argument('--trigger_dir', type=str, required=True, 
                        help="Directory containing the saved 'adaptive_mask.png' and 'adaptive_pattern.png' from adaptive_attack.py.")

    # --- Model & Dataset ---
    parser.add_argument('--model', type=str, required=True, 
                        choices=['resnet18', 'resnet50', 'vgg16', 'vit'],
                        help="Architecture of the robust model.")
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['cifar10', 'cifar100', 'gtsrb', 'tinyimagenet'],
                        help="Dataset the model was trained on.")

    # --- System Arguments ---
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help="Disable CUDA.")
    
    args = parser.parse_args()
    evaluate_with_saved_trigger(args)

