# coding: utf-8
"""
Evaluate a Pre-Generated Trigger

This script loads a model and a pre-generated trigger (as a .npy file) 
and evaluates its performance on the clean test set.

It reports three key metrics:
1.  Clean Accuracy: The model's baseline accuracy on clean, unmodified data.
2.  Reversed Accuracy: The accuracy on triggered data, measured against the
    *original* true labels (i.e., "utility preservation").
3.  ASR (Attack Success Rate): The accuracy on triggered data, measured against
    a single *target* class (if --target_class is provided).

This script is hard-coded for ResNet18 and .npy data files.
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import argparse
import numpy as np
import sys
import os

current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
root_dir = os.path.dirname(current_dir)

if root_dir not in sys.path:
    sys.path.append(root_dir)
from Models.resnet import ResNet18
import random
import time
import torch
import torch.utils.data as data

from torchvision import transforms as T

def get_model(model_path, num_classes):
    """Loads a ResNet18 model from a .pth file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = ResNet18(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    print(f"Successfully loaded model weights from '{model_path}'.")
    return model

def get_norm():
    """Returns the CIFAR-10 normalization transform."""
    # Note: These are standard CIFAR-10 means, but the std dev
    # (0.2470, 0.2435, 0.2616) is different from the typical (0.2023, 0.1994, 0.2010).
    mean = torch.FloatTensor([0.4914, 0.4822, 0.4465])
    std  = torch.FloatTensor([0.2470, 0.2435, 0.2616])
    normalize = T.Normalize(mean, std)
    return normalize

def preprocess(inputs):
    """Converts NumPy data (NHWC, uint8) to FloatTensor (NCHW, 0-1)."""
    if inputs.dtype == np.uint8:
        inputs = inputs.astype('float32')
    inputs = np.transpose(inputs / 255.0, [0, 3, 1, 2])
    inputs = torch.FloatTensor(inputs)
    return inputs

def main():
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu != 'cpu' else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # --- 1. Initialization and Loading ---
    print("--- Initializing model, data, and trigger ---")
    model = get_model(args.model_path, args.num_classes).to(args.device)
    model.eval()
    normalize = get_norm()

    # Load the pre-generated trigger
    if not os.path.exists(args.trigger_path):
        raise FileNotFoundError(f"Trigger file not found: {args.trigger_path}")
    pattern = torch.from_numpy(np.load(args.trigger_path)).to(args.device)
    print(f"Successfully loaded trigger from '{args.trigger_path}'.")
    
    # Load CIFAR-10 test data
    x_test = np.load('data/cifar10_test_x.npy')
    y_test_true = np.load('data/cifar10_test_y.npy')
    x_test_processed = preprocess(x_test)
    y_test_true_processed = torch.LongTensor(y_test_true)
    
    # Create DataLoader for batch processing
    test_dataset = data.TensorDataset(x_test_processed, y_test_true_processed)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    total_samples = len(test_dataset)
    print(f"Loaded {total_samples} test samples.")

    # --- 2. Evaluating Baseline Clean Accuracy ---
    print("\n--- 2. Evaluating Baseline Clean Accuracy ---")
    correct_clean = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
            output = model(normalize(x_batch))
            pred = output.argmax(dim=1)
            correct_clean += (pred == y_batch).sum().item()
    
    clean_accuracy = correct_clean / total_samples
    print(f"Clean Model Accuracy: {clean_accuracy:.4f} ({correct_clean}/{total_samples})")

    # --- 3. Evaluating Performance with Trigger ---
    print(f"\n--- 3. Evaluating Performance with Trigger ---")
    correct_reversed = 0
    correct_to_target = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
            
            # Apply trigger
            x_adv = torch.clamp(x_batch + pattern, 0, 1)
            output = model(normalize(x_adv))
            pred = output.argmax(dim=1)
            
            # Calculate reversed accuracy (original label preservation)
            correct_reversed += (pred == y_batch).sum().item()
            
            # If target class is specified, calculate ASR
            if args.target_class is not None:
                correct_to_target += (pred == args.target_class).sum().item()
    
    # --- 4. Print Final Results ---
    print("\n" + "="*50)
    print(" " * 18 + "Final Evaluation Results")
    print("="*50)
    print(f"Baseline Accuracy (Clean):         {clean_accuracy:.4f}")

    reversed_accuracy = correct_reversed / total_samples
    print(f"Reversed Accuracy (Triggered): {reversed_accuracy:.4f}")
    
    if args.target_class is not None:
        asr = correct_to_target / total_samples
        print(f"Attack Success Rate (ASR to class {args.target_class}):       {asr:.4f}")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a recovered trigger on a model.')
    
    # --- Required Arguments ---
    parser.add_argument('--model_path',  required=True,      
                        help='Path to your trained .pth model file')
    parser.add_argument('--trigger_path',required=True,      
                        help='Path to the recovered trigger .npy file')
    
    # --- Optional Arguments ---
    parser.add_argument('--target_class',type=int,           
                        help='(Optional) The target class the trigger was designed for, to calculate ASR.')
    parser.add_argument('--gpu',         default='0',        
                        help='GPU ID to use (e.g., "0" or "cpu")')
    parser.add_argument('--batch_size',  default=256,  type=int, 
                        help='Batch size for evaluation')
    parser.add_argument('--num_classes', default=10,   type=int, 
                        help='Number of classes for the dataset (e.g., 10 for CIFAR-10)')


    main()
