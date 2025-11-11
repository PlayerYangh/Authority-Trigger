"""
Adaptive Trigger-Reversal Attack Script

This script implements an adaptive attack to evaluate the robustness of a 
benign backdoor model. The goal of this attack is not to cause 
misclassification to a target class, but rather to "unlock" the model's 
intended functionality on clean data.

The attack works by optimizing a trigger (a 'pattern' and a 'mask') that, 
when applied to clean images, maximizes the model's classification accuracy 
on those images' *true* labels.

The optimization is constrained by the L1 norm of the mask to find the 
smallest (most sparse) trigger possible, simulating an attacker's goal
of creating an inconspicuous trigger.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os
import argparse
from tqdm import tqdm
from torchvision.datasets.folder import IMG_EXTENSIONS

from resnet import ResNet18, ResNet50
from vgg import VGG16_CIFAR100
from vit import create_vit

from utils import AverageMeter as Average, Accuracy

def get_dataset_params(dataset_name):
    """
    Returns normalization stats, class count, and crop size for a given dataset.
    """
    if dataset_name == 'cifar10': 
        mean, std, num_classes, crop_size = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 10, 32
    elif dataset_name == 'cifar100': 
        mean, std, num_classes, crop_size = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 100, 32
    elif dataset_name == 'gtsrb': 
        mean, std, num_classes, crop_size = (0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669), 43, 32
    elif dataset_name == 'tinyimagenet': 
        mean, std, num_classes, crop_size = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 200, 64
        # Note: Override to 32 to match the main training script's default resize
        crop_size = 32
    else: 
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return mean, std, num_classes, crop_size

class UnNormalize(object):
    """Reverses the T.Normalize transformation."""
    def __init__(self, mean, std):
        self.unnormalize = T.Normalize(
            mean=[-m/s for m, s in zip(mean, std)],
            std=[1/s for s in std]
        )
    def __call__(self, tensor):
        return self.unnormalize(tensor)

def save_final_trigger(pattern, mask, save_dir, prefix=""):
    """
    Saves the optimized pattern, mask, and their fusion as images.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Detach from graph and move to CPU
    pattern_cpu = pattern.detach().cpu()
    mask_cpu = mask.detach().squeeze(0).cpu()
    
    # Convert to image format (0-255 uint8)
    mask_img = (mask_cpu.numpy() * 255).astype(np.uint8)
    pattern_img = (pattern_cpu.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # Create a 3-channel fusion image (mask * pattern)
    fusion_mask = mask_cpu.repeat(3, 1, 1).permute(1, 2, 0).numpy()
    fusion_img = (fusion_mask * pattern_img).astype(np.uint8)
    
    # Save all three images
    Image.fromarray(mask_img, mode='L').save(os.path.join(save_dir, f"{prefix}adaptive_mask.png"))
    Image.fromarray(pattern_img).save(os.path.join(save_dir, f"{prefix}adaptive_pattern.png"))
    Image.fromarray(fusion_img).save(os.path.join(save_dir, f"{prefix}adaptive_fusion.png"))
    
    print(f"Adaptive attack results saved in '{save_dir}'")


def adaptive_attack(args):
    """Main function to run the adaptive trigger-reversal attack."""
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # --- [FIX] Add support for .JPEG extension (for Tiny ImageNet) ---
    global IMG_EXTENSIONS
    if '.JPEG' not in IMG_EXTENSIONS:
        IMG_EXTENSIONS = IMG_EXTENSIONS + ('.JPEG',)
    # --- [END FIX] ---

    mean, std, num_classes, crop_size = get_dataset_params(args.dataset)

    # 1. Load the victim model
    model_path = args.model_path
    if args.model == 'resnet18': net = ResNet18(num_classes=num_classes)
    elif args.model == 'resnet50': net = ResNet50(num_classes=num_classes)
    elif args.model == 'vgg16': net = VGG16_CIFAR100(num_classes=num_classes)
    elif args.model == 'vit':
        net = create_vit(dataset_name=args.dataset, num_classes=num_classes)
    else: 
        raise ValueError(f"Unsupported model: {args.model}")

    if not os.path.exists(model_path): 
        raise FileNotFoundError(f"Model file not found at {model_path}")
    print(f"Loading victim model from: {model_path}")
    # Use weights_only=True for safer loading
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    net.eval()
    net.to(device)

    # 2. Load clean data for optimization
    transform_test = T.Compose([T.Resize((crop_size, crop_size)), T.ToTensor(), T.Normalize(mean, std)])
    # We use the *clean* test set (or a subset) as the basis for optimization
    clean_dataset = ImageFolder(root=args.clean_data_dir, transform=transform_test)
    clean_loader = DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"Loaded {len(clean_dataset)} clean images from '{args.clean_data_dir}' for optimization.")

    # 3. Initialize trigger (pattern and mask)
    img_shape = (3, crop_size, crop_size)
    # Use Tanh space for optimization (values from -inf to +inf)
    # We will pass these through (tanh(x) + 1) / 2 to get values in [0, 1]
    mask_tanh = torch.zeros(1, img_shape[1], img_shape[2], device=device, requires_grad=True)
    pattern_tanh = torch.zeros(img_shape, device=device, requires_grad=True)
    # Initialize with small random values
    nn.init.xavier_uniform_(mask_tanh)
    nn.init.xavier_uniform_(pattern_tanh)

    # 4. Setup optimizer and loss
    if args.model == 'vit':
        print("Using AdamW optimizer for ViT model.")
        optimizer = optim.AdamW([mask_tanh, pattern_tanh], lr=args.learning_rate_vit, weight_decay=1e-4)
    else: # CNNs
        print("Using Adam optimizer for CNN model.")
        optimizer = optim.Adam([mask_tanh, pattern_tanh], lr=args.learning_rate)
        
    criterion = nn.CrossEntropyLoss()
    cost = args.l1_cost_init  # Initial weight for the L1 sparsity loss
    
    best_acc = 0.0
    best_mask, best_pattern = None, None
    best_l1_norm = float('inf')

    # Create helpers to normalize and un-normalize images
    unnormalize = UnNormalize(mean, std)
    normalize = T.Normalize(mean, std)
    
    # 5. Optimization Loop
    for epoch in range(args.epochs):
        epoch_loss, epoch_acc, epoch_l1 = Average(), Accuracy(), Average()
        
        for images_norm, true_labels in tqdm(clean_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images_norm, true_labels = images_norm.to(device), true_labels.to(device)
            
            # (a) Get current mask and pattern in [0, 1] range
            mask = torch.tanh(mask_tanh).add(1).mul(0.5)
            pattern = torch.tanh(pattern_tanh).add(1).mul(0.5)
            
            # (b) Apply the trigger
            # Un-normalize images to pixel space [0, 1]
            images_unnorm = unnormalize(images_norm)
            # Apply trigger: (1-M) * X + M * P
            triggered_images_unnorm = (1 - mask) * images_unnorm + mask * pattern
            # Clamp to valid pixel range
            triggered_images_unnorm = torch.clamp(triggered_images_unnorm, 0.0, 1.0)
            # Re-normalize for model input
            triggered_images_norm = normalize(triggered_images_unnorm)
            
            optimizer.zero_grad()
            output = net(triggered_images_norm)
            
            # (c) Calculate composite loss
            # Loss 1: Classification loss (we want to maximize this)
            loss_ce = criterion(output, true_labels)
            # Loss 2: L1 norm of the mask (we want to minimize this)
            loss_l1 = torch.norm(mask, p=1)
            # Total loss: L = L_class + cost * L_1
            total_loss = loss_ce + cost * loss_l1
            
            total_loss.backward()
            optimizer.step()
            
            # (d) Update metrics
            epoch_acc.update(output, true_labels)
            epoch_loss.update(total_loss.item(), images_norm.size(0))
            epoch_l1.update(loss_l1.item(), 1) # Avg L1 norm per batch
            
        avg_acc = epoch_acc.accuracy
        avg_l1 = epoch_l1.average
        
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss.average:.4f}, "
              f"Recovery Acc: {avg_acc*100:.2f}%, Mask L1 Norm: {avg_l1:.2f}, Cost: {cost:.6f}")
        
        # (e) Dynamically adjust the cost (L1 weight)
        if avg_acc > args.acc_success_threshold:
            # If accuracy is good, we can focus on minimizing the mask
            if avg_l1 > args.target_l1_norm:
                cost *= args.cost_multiplier_up
                print(f"  Info: Acc high, L1 high. Increasing cost sharply to {cost:.6f}")
            else:
                cost *= args.cost_multiplier_down
                print(f"  Info: Acc and L1 good. Increasing cost gently to {cost:.6f}")
        else:
            # If accuracy is low, we need to reduce the L1 penalty
            cost /= args.cost_multiplier_up
            print(f"  Info: Acc low. Decreasing cost to {cost:.6f}")
        
        cost = max(cost, 1e-5) # Prevent cost from becoming too small
        
        # (f) Save the best trigger found so far
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_mask = torch.tanh(mask_tanh).add(1).mul(0.5).detach()
            best_pattern = torch.tanh(pattern_tanh).add(1).mul(0.5).detach()
            best_l1_norm = torch.norm(best_mask, p=1).item()

    # 6. Final Results
    print("\n--- Adaptive Attack Finished ---")
    if best_mask is not None and best_pattern is not None:
        fusion_tensor = best_mask.repeat(3, 1, 1) * best_pattern
        l2_norm_of_trigger = torch.linalg.norm(fusion_tensor).item()
        
        print(f"Best Recovery Accuracy Achieved: {best_acc*100:.2f}%")
        print(f"L1 Norm of the best mask: {best_l1_norm:.2f}")
        print(f"L2 Norm of the best trigger (fusion): {l2_norm_of_trigger:.4f}")
        
        if best_l1_norm <= args.l1_norm_constraint:
            print(f"Result: SUCCESSFUL - Found a trigger with L1 norm ({best_l1_norm:.2f}) "
                  f"<= constraint ({args.l1_norm_constraint})")
        else:
            print(f"Result: FAILED - The best trigger's L1 norm ({best_l1_norm:.2f}) "
                  f"> constraint ({args.l1_norm_constraint})")
            
        # Save the resulting trigger images
        save_final_trigger(best_pattern, best_mask, args.save_dir, 
                           prefix=f"acc_{best_acc*100:.1f}_l1_{best_l1_norm:.1f}_")
    else:
        print("No effective trigger was found during optimization.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adaptive attack against benign backdoors with a constrained optimization approach.")
    
    # --- Path Arguments ---
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the pre-trained victim model (.pth).")
    parser.add_argument('--clean_data_dir', type=str, required=True,
                        help="Path to the *clean* test dataset (ImageFolder format).")
    
    # --- Model & Dataset ---
    parser.add_argument('--model', type=str, required=True, 
                        choices=['resnet18', 'resnet50', 'vgg16', 'vit'],
                        help="Victim model architecture.")
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['cifar10', 'cifar100', 'gtsrb', 'tinyimagenet'],
                        help="Dataset the model was trained on.")

    # --- Optimization Hyperparameters ---
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.05, 
                        help='Initial learning rate for CNN models (Adam).')
    parser.add_argument('--learning_rate_vit', type=float, default=1e-4, 
                        help='Initial learning rate for ViT (AdamW).')
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of optimization epochs.")
    parser.add_argument('--batch_size', type=int, default=64)
    
    # --- Cost Function Dynamics ---
    parser.add_argument('--l1_cost_init', type=float, default=1e-2,
                        help="Initial weight for the L1 sparsity cost.")
    parser.add_argument('--target_l1_norm', type=float, default=15.0,
                        help="L1 norm threshold to aim for.")
    parser.add_argument('--acc_success_threshold', type=float, default=0.90,
                        help="Accuracy threshold to consider the attack 'successful' for cost adjustment (e.g., 0.9 for 90%%).")
    parser.add_argument('--cost_multiplier_up', type=float, default=2.0,
                        help="Factor to increase L1 cost by.")
    parser.add_argument('--cost_multiplier_down', type=float, default=1.1,
                        help="Factor to gently increase L1 cost by.")
    
    # --- Misc. Arguments ---
    parser.add_argument('--l1_norm_constraint', type=float, default=49.0,
                        help="Final L1 norm to consider the attack a 'SUCCESS' (e.g., 7x7=49 pixels).")
    parser.add_argument('--save_dir', type=str, default='adaptive_attack_results',
                        help="Directory to save the generated trigger images.")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    
    args = parser.parse_args()
    adaptive_attack(args)