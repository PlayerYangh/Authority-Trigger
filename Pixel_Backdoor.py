# coding: utf-8
"""
Trigger Inversion Attack Scanner (for traditional ASR)

This script performs an automated scan to evaluate a model's vulnerability 
to traditional backdoor attacks (i.e., forcing classification to a specific target class).

It iterates through *every* possible class in the dataset (e.g., 0-9 for CIFAR-10)
and treats each one as a potential 'target_class'.

For each target class, it uses the 'PixelBackdoor' inversion algorithm 
from the 'inversion_torch' library to generate an optimal trigger.

It then evaluates and reports two key metrics for that trigger:
1.  ASR (Attack Success Rate): The percentage of test images that are 
    misclassified as the 'target_class' when the trigger is applied.
2.  Reversed Acc: The percentage of test images that are *still* correctly
    classified (to their original, true label) even with the trigger applied.
    
This script depends on the 'inversion_torch.py' library and pre-processed
NumPy data files ('data/cifar10_train_x.npy', etc.).
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import argparse
import numpy as np
import os
import random
import time
import torch

from torchvision import transforms as T

# This script is hard-coded for ResNet18 and requires the inversion_torch library
from resnet import ResNet18
from inversion_torch import PixelBackdoor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
    # This must match the 'inversion_torch' library's requirements.
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

def evaluate_all_classes():
    """
    Main function to loop through all classes, generate a trigger for each,
    and evaluate its ASR and Reversed Accuracy.
    """
    # --- 1. Initialization (Load model and data once) ---
    print("--- 1. Initializing Model and Data ---")
    model = get_model(args.model_path, args.num_classes).to(args.device)
    model.eval()
    normalize = get_norm()
    
    # Load data for trigger generation
    x_val = np.load('data/cifar10_train_x.npy')
    y_val = np.load('data/cifar10_train_y.npy')
    x_val_processed = preprocess(x_val)
    y_val_processed = torch.LongTensor(y_val)
    print(f'Generation set shape: {x_val.shape}')

    # Load data for evaluation
    x_test = np.load('data/cifar10_test_x.npy')
    y_test_true = np.load('data/cifar10_test_y.npy')
    x_test_processed = preprocess(x_test).to(args.device)
    y_test_true_processed = torch.LongTensor(y_test_true).to(args.device)
    print(f'Test set shape: {x_test.shape}')

    # List to store all results
    results = []

    # --- 2. Iterate Through All Target Classes ---
    for target_class in range(args.num_classes):
        print(f"\n{'='*25} Scanning Target Class: {target_class} {'='*25}")
        # Use 'all-to-one' attack mode
        source_class = args.num_classes  

        # Initialize trigger inversion tool
        backdoor = PixelBackdoor(model,
                                 shape=(3, 32, 32),
                                 num_classes=args.num_classes,
                                 batch_size=args.batch_size,
                                 normalize=normalize)

        # Generate trigger
        print(f"Generating trigger for target class {target_class}...")
        pattern = backdoor.generate((source_class, target_class),
                                    x_val_processed,
                                    y_val_processed,
                                    attack_size=args.attack_size)

        trigger_size = np.count_nonzero(pattern.abs().sum(0).cpu().numpy())
        print(f"Target {target_class} | Trigger Size: {trigger_size}")

        # Evaluate trigger effectiveness
        with torch.no_grad():
            x_adv = torch.clamp(x_test_processed + pattern.to(args.device), 0, 1)
            x_adv = normalize(x_adv)
            pred = model(x_adv).argmax(dim=1)
        
        # Calculate ASR (Attack Success Rate)
        asr = (pred == target_class).sum().item() / pred.size(0)
        # Calculate Reversed Acc (Original Label Preservation)
        reversed_acc = (pred == y_test_true_processed).sum().item() / pred.size(0)

        print(f"Target {target_class} -> ASR: {asr:.4f}, Reversed Acc: {reversed_acc:.4f}")
        
        # Store results for this class
        results.append({
            "target_class": target_class,
            "trigger_size": trigger_size,
            "asr": asr,
            "reversed_acc": reversed_acc
        })

    # --- 3. Final Summary Report ---
    print("\n\n" + "="*80)
    print(" " * 28 + "Automated Scan Final Results")
    print("="*80)
    print(f"{'Target Class':<15} | {'Trigger Size':<15} | {'ASR':<20} | {'Reversed Accuracy':<20}")
    print("-"*80)
    for res in results:
        print(f"{res['target_class']:<15} | {res['trigger_size']:<15} | {res['asr']:<20.4f} | {res['reversed_acc']:<20.4f}")
    print("="*80)


################################################################
############                  main                  ############
################################################################
def main():
    # This script is designed to run the full scan
    if args.phase == 'scan':
        evaluate_all_classes()
    else:
        print(f"This script currently only supports 'scan' mode. You provided: '{args.phase}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a full scan for trigger-reversal (ASR) attacks.')
    
    # --- Required Arguments ---
    parser.add_argument('--model_path',  required=True,      
                        help='Path to your trained .pth model file (e.g., resnet18_cifar10_model_A.pth)')

    # --- Script Configuration ---
    parser.add_argument('--gpu',         default='0',        help='GPU ID to use (e.g., "0" or "cpu")')
    parser.add_argument('--phase',       default='scan',     
                        help='Phase of framework, only "scan" is supported to run all classes')
    parser.add_argument('--seed',        default=1024, type=int, help='Random seed')
    parser.add_argument('--batch_size',  default=128,  type=int, help='Batch size for inversion and evaluation')
    parser.add_argument('--num_classes', default=10,   type=int, help='Number of classes for the dataset (e.g., 10 for CIFAR-10)')
    parser.add_argument('--attack_size', default=50,   type=int, 
                        help='Number of samples from the generation set to use for inversion')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    time_start = time.time()
    main()
    time_end = time.time()
    print('='*50)
    print(f'Total Running time: {(time_end - time_start) / 60:.4f} m')
    print('='*50)