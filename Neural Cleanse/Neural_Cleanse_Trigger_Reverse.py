#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Neural Cleanse Trigger Generation Script (Step 1)

This script implements the first step of the Neural Cleanse attack:
Iterating through all possible target classes and optimizing to find the 
smallest possible trigger (pattern + mask) that induces a misclassification
to that target class (ASR).

It saves the resulting pattern and mask images for each class to a
specified results directory. These masks are then analyzed by the
'analyze_cleanse_results.py' script.

This script is based on the original Neural Cleanse implementation and
is configured to work with ResNet18 on CIFAR-10 or GTSRB.
It requires the 'visualizer_pytorch.py' and 'utils_backdoor_pytorch.py'
utility files.
"""

import os
import time
import numpy as np
import random
import argparse

import torch
import torch.nn as nn
from resnet import ResNet18 # Hard-coded for ResNet18
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# --- Neural Cleanse Lib Imports ---
from visualizer_pytorch import Visualizer 
import utils_backdoor_pytorch as utils_backdoor 

# Set random seeds for reproducibility
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)
random.seed(123)
np.random.seed(123)


def get_dataset_params(dataset_name):
    """Returns params for CIFAR-10 or GTSRB."""
    if dataset_name == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        num_classes = 10
        input_shape = (32, 32, 3) # (H, W, C)
    elif dataset_name == 'gtsrb':
        # Using common GTSRB stats, adjust if yours differ
        mean, std = (0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)
        num_classes = 43
        input_shape = (32, 32, 3) # (H, W, C)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return mean, std, num_classes, input_shape


def load_data_for_visualizer(data_dir, dataset_name):
    """
    Loads data from ImageFolder and converts to the NumPy (N, H, W, C)
    format required by the Visualizer library.
    """
    
    mean, std, num_classes, input_shape = get_dataset_params(dataset_name)
    
    transform_test = transforms.Compose([
        transforms.Resize((input_shape[0], input_shape[1])), # H, W
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if not os.path.exists(data_dir):
         raise FileNotFoundError(f"Test data path {data_dir} not found.")

    print(f"Loading {dataset_name} image data from: {data_dir}")
    testset = datasets.ImageFolder(root=data_dir, transform=transform_test)
    
    X_test_list = []
    Y_test_list = []
    
    for img_tensor, label in testset:
        # img_tensor is (C, H, W)
        # Visualizer expects (H, W, C) NumPy array
        img_np = img_tensor.permute(1, 2, 0).numpy() # (H, W, C)
        X_test_list.append(img_np) 
        Y_test_list.append(label)

    X_test = np.array(X_test_list, dtype='float32')
    Y_test = np.array(Y_test_list, dtype='int64')

    print(f'X_test shape (N, H, W, C): {X_test.shape}')
    print(f'Y_test shape (N,): {Y_test.shape}')
    
    return X_test, Y_test, num_classes, input_shape


def visualize_trigger_w_mask(visualizer, data_loader, y_target, input_shape,
                             result_dir, img_filename_template,
                             args):
    """
    Core function to run the optimization for a single target class.
    """
    visualize_start_time = time.time()

    # initialize with random mask
    pattern = np.random.random(input_shape) * 255.0
    mask_h, mask_w = int(np.ceil(input_shape[0] / args.upsample_size)), \
                     int(np.ceil(input_shape[1] / args.upsample_size))
    mask = np.random.random((mask_h, mask_w))

    pattern, mask, mask_upsample, logs = visualizer.visualize(
        data_loader=data_loader, y_target=y_target, pattern_init=pattern, mask_init=mask)

    print(f'Pattern shape: {pattern.shape}, min: {np.min(pattern):.3f}, max: {np.max(pattern):.3f}')
    print(f'Mask shape: {mask.shape}, min: {np.min(mask):.3f}, max: {np.max(mask):.3f}')
    print(f'Mask L1 norm for label {y_target}: {np.sum(np.abs(mask_upsample)):.4f}')

    visualize_end_time = time.time()
    print(f'Visualization cost: {(visualize_end_time - visualize_start_time):.2f} seconds')

    if args.save_pattern:
        save_pattern(pattern, mask_upsample, y_target, result_dir, img_filename_template)

    return pattern, mask, mask_upsample, logs


def save_pattern(pattern, mask_upsample, y_target, 
                 result_dir, img_filename_template):
    """Saves the generated pattern, mask, and fusion images."""
    
    os.makedirs(result_dir, exist_ok=True)

    img_filename = os.path.join(result_dir, img_filename_template % ('pattern', y_target))
    utils_backdoor.dump_image(pattern, img_filename, 'png')

    img_filename = os.path.join(result_dir, img_filename_template % ('mask', y_target))
    utils_backdoor.dump_image(np.expand_dims(mask_upsample, axis=2) * 255.0,
                              img_filename,
                              'png')

    fusion = np.multiply(pattern, np.expand_dims(mask_upsample, axis=2))
    img_filename = os.path.join(result_dir, img_filename_template % ('fusion', y_target))
    utils_backdoor.dump_image(fusion, img_filename, 'png')
    
    print(f"Saved pattern/mask/fusion for target {y_target} to {result_dir}")


def run_scan(args):
    """
    Main execution function: loads data, loads model, and scans all target classes.
    """
    
    print('Loading dataset...')
    X_test_np, Y_test_np, num_classes, input_shape = load_data_for_visualizer(
        args.data_dir, args.dataset
    )

    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
    Y_test_tensor = torch.tensor(Y_test_np, dtype=torch.long)

    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print('Loading model...')
    # This script is hard-coded for ResNet18
    model = ResNet18(num_classes=num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)
    model.eval()
    print("Model loaded successfully.")

    # --- Initialize Visualizer ---
    # Calculate mini-batch size
    if args.nb_sample % args.batch_size != 0:
        raise ValueError("nb_sample must be divisible by batch_size.")
    mini_batch_size = args.nb_sample // args.batch_size

    visualizer = Visualizer(
        model=model, 
        intensity_range=args.intensity_range, 
        regularization=args.regularization,
        input_shape=input_shape, # (H, W, C)
        init_cost=args.init_cost, 
        steps=args.steps, 
        lr=args.lr, 
        num_classes=num_classes,
        mini_batch=mini_batch_size,
        upsample_size=args.upsample_size,
        attack_succ_threshold=args.attack_succ_threshold,
        patience=args.patience, 
        cost_multiplier=args.cost_multiplier,
        img_color=input_shape[2], 
        batch_size=args.batch_size, 
        verbose=2,
        save_last=args.save_last,
        early_stop=args.early_stop, 
        early_stop_threshold=args.early_stop_threshold,
        early_stop_patience=args.early_stop_patience,
        raw_input_flag=True, # We are feeding raw, normalized tensors
        device=args.device
    )

    log_mapping = {}

    y_target_list = list(range(num_classes))
    if args.target_class is not None and args.target_class in y_target_list:
        y_target_list.remove(args.target_class)
        y_target_list = [args.target_class] + y_target_list
    
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Define the template for saving images
    img_filename_template = f"{args.dataset}_resnet18_visualize_%s_label_%d.png"

    # --- Run Scan ---
    for y_target in y_target_list:
        print(f"\n--- Processing Target Class {y_target} ---")
        _, _, _, logs = visualize_trigger_w_mask(
            visualizer, test_loader, y_target=y_target,
            input_shape=input_shape,
            result_dir=args.result_dir,
            img_filename_template=img_filename_template,
            args=args
        )
        log_mapping[y_target] = logs


def main():
    parser = argparse.ArgumentParser(description="Run Neural Cleanse trigger generation for all classes.")
    
    # --- Essential Arguments ---
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['cifar10', 'gtsrb'],
                        help="Dataset name (cifar10 or gtsrb).")
    parser.add_argument('--model_path', type=str, required=True, 
                        help="Path to the pre-trained ResNet18 .pth model file.")
    parser.add_argument('--data_dir', type=str, required=True, 
                        help="Path to the clean *test* dataset (ImageFolder format).")
    parser.add_argument('--result_dir', type=str, default='neural_cleanse_results', 
                        help="Directory to save the generated trigger/mask images.")
    
    # --- Scan & Optimization Arguments ---
    parser.add_argument('--target_class', type=int, default=None, 
                        help="(Optional) Specify a single class to run first.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for evaluation (must be <= nb_sample).")
    parser.add_argument('--lr', type=float, default=0.1, 
                        help="Learning rate for trigger optimization.")
    parser.add_argument('--steps', type=int, default=100, 
                        help="Number of optimization steps per class.")
    parser.add_argument('--nb_sample', type=int, default=100,
                        help="Number of samples to use for optimization (must be >= batch_size).")
    parser.add_argument('--init_cost', type=float, default=1e-3,
                        help="Initial cost for L1 regularization.")
    parser.add_argument('--regularization', type=str, default='l1', 
                        help="Regularization type (default: l1).")
    parser.add_argument('--attack_succ_threshold', type=float, default=0.99,
                        help="ASR threshold to consider the attack successful.")
    parser.add_argument('--patience', type=int, default=5,
                        help="Patience steps for cost multiplier adjustment.")
    parser.add_argument('--cost_multiplier', type=float, default=2.0,
                        help="Factor to increase cost by when ASR is high.")
    parser.add_argument('--save_pattern', action='store_true', default=True,
                        help="Save the generated trigger/mask images.")
    
    # --- Other Arguments ---
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use (or "cpu").')
    parser.add_argument('--intensity_range', type=str, default='raw', 
                        help="Input range. 'raw' means data is already preprocessed (e.g., normalized).")
    parser.add_argument('--upsample_size', type=int, default=1,
                        help="Upsampling factor for the mask.")
    parser.add_argument('--save_last', action='store_true',
                        help="Save the trigger from the last step, not the best one.")
    parser.add_argument('--early_stop', action='store_true', default=True,
                        help="Enable early stopping.")
    parser.add_argument('--early_stop_threshold', type=float, default=1.0,
                        help="ASR threshold for early stopping (e.g., 1.0 for 100%%).")
    parser.add_argument('--early_stop_patience', type=int, default=25,
                        help="Patience for early stopping.")

    args = parser.parse_args()
    
    args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu != 'cpu' else "cpu")
    print(f"Using device: {args.device}")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.gpu != 'cpu' else ""

    start_time = time.time()
    run_scan(args)
    elapsed_time = time.time() - start_time
    print(f'Total elapsed time: {elapsed_time:.2f} s')


if __name__ == '__main__':
    main()
