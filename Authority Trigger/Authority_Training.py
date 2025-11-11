"""
Main training script for the Authority Trigger model.

This script implements the core training logic for the paper, 
combining three distinct datasets with a composite loss function
to create a robust and finetune-resistant benign backdoor.
"""

from __future__ import division, print_function

import argparse
from itertools import cycle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
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
from Common.utils import AverageMeter, Accuracy

class SourceTaggedDataset(torch.utils.data.Dataset):
    """
    A wrapper dataset that adds a 'source_id' to each sample.
    This is used to tag samples from D_auth (0), D_rand (1), and D_noise (2)
    before concatenating them into a single dataset.
    """
    def __init__(self, dataset, source_id):
        self.dataset = dataset
        self.source_id = source_id
    def __getitem__(self, index):
        data, label = self.dataset[index]
        # Return data, label, and the source ID
        return data, label, self.source_id
    def __len__(self):
        return len(self.dataset)

class Trainer(object):
    """Handles the main model training and evaluation loop."""
    def __init__(self, net, optimizer, train_loader, l_ft_loader,
                 test_loader_triggered, test_loader_clean_correct,
                 device, scheduler, model_save_prefix,
                 lambda_d2_weight, theta_d_c_weight, gamma_l_ft,
                 clean_acc_threshold, triggered_acc_threshold, noise_sd):
        
        self.net = net
        self.optimizer = optimizer
        self.train_loader = train_loader  
        self.l_ft_loader = l_ft_loader    # Loader for finetune-resistance (clean data)
        self.test_loader_triggered = test_loader_triggered
        self.test_loader_clean_correct = test_loader_clean_correct
        self.device = device
        
        # Loss function for per-sample weighting
        self.loss_func_no_reduction = torch.nn.CrossEntropyLoss(reduction='none')
        # Standard loss function for other tasks
        self.loss_func_mean_reduction = torch.nn.CrossEntropyLoss()
        
        self.scheduler = scheduler
        self.model_save_prefix = model_save_prefix
        
        # Loss weights
        self.lambda_d2_weight = lambda_d2_weight  # (lambda) for D_rand
        self.theta_d_c_weight = theta_d_c_weight  # (theta) for D_noise
        self.gamma_l_ft = gamma_l_ft            # (gamma) for finetune-resistance
        
        self.noise_sd = noise_sd # Std dev for randomized smoothing
        
        # Thresholds for saving complex models (C and D)
        self.clean_acc_threshold_C = clean_acc_threshold
        self.triggered_acc_threshold_D = triggered_acc_threshold

        # State variables for saving the four best models
        self.best_acc_triggered_val = 0.0
        self.best_epoch_triggered = 0
        self.last_saved_triggered_clean_acc = 0.0
        
        self.lowest_clean_correct_acc_val = float('inf')
        self.best_epoch_lowest_clean = 0
        self.lowest_clean_corresponding_triggered_acc = 0.0
        
        self.best_comp_C_triggered_acc = 0.0
        self.best_comp_C_clean_acc = float('inf')
        self.best_epoch_C = 0
        
        self.best_comp_D_clean_acc = float('inf')
        self.best_comp_D_triggered_acc = 0.0
        self.best_epoch_D = 0


    def fit(self, epochs, dataset_name):
        """
        Runs the main training loop for a specified number of epochs.
        Handles training, evaluation, and model checkpointing.
        """
        self.dataset_name = dataset_name
        print(f"Starting training for {epochs} epochs...")
        print(f"  - D_rand loss weight (lambda): {self.lambda_d2_weight}")
        print(f"  - D_noise loss weight (theta): {self.theta_d_c_weight}")
        print(f"  - Finetune-resist loss weight (gamma): {self.gamma_l_ft}")
        print(f"  - Gaussian noise augmentation std (sigma): {self.noise_sd}")
        print(f"Model C condition: Clean Acc < {self.clean_acc_threshold_C*100:.2f}%")
        print(f"Model D condition: Triggered Acc > {self.triggered_acc_threshold_D*100:.2f}%")
        print(f"Saving models with prefix: {self.model_save_prefix}")

        for epoch in range(1, epochs + 1):
            train_loss_avg, train_acc_obj = self.train()
            
            triggered_loss_avg, triggered_acc_obj = self.evaluate(self.test_loader_triggered)
            clean_correct_loss_avg, clean_correct_acc_obj = self.evaluate(self.test_loader_clean_correct)
            
            current_triggered_acc = triggered_acc_obj.accuracy
            current_clean_correct_acc = clean_correct_acc_obj.accuracy

            # --- Model Checkpointing Logic ---
            # Model A: Best Triggered (Authorized) Accuracy
            if current_triggered_acc > self.best_acc_triggered_val:
                print(f"\n-- Update Model A (Best Triggered) --")
                self.best_acc_triggered_val, self.last_saved_triggered_clean_acc, self.best_epoch_triggered = current_triggered_acc, current_clean_correct_acc, epoch
                torch.save(self.net.state_dict(), f"{self.model_save_prefix}_model_A_best_triggered.pth")
                print(f"New Model A saved at epoch {epoch}! Triggered Acc: {self.best_acc_triggered_val*100:.2f}% (Clean Acc: {self.last_saved_triggered_clean_acc*100:.2f}%)")

            # Model B: Lowest Clean (Unauthorized) Accuracy
            if current_clean_correct_acc < self.lowest_clean_correct_acc_val:
                print(f"\n-- Update Model B (Lowest Clean) --")
                self.lowest_clean_correct_acc_val, self.lowest_clean_corresponding_triggered_acc, self.best_epoch_lowest_clean = current_clean_correct_acc, current_triggered_acc, epoch
                torch.save(self.net.state_dict(), f"{self.model_save_prefix}_model_B_lowest_clean.pth")
                print(f"New Model B saved at epoch {epoch}! Clean Acc: {self.lowest_clean_correct_acc_val*100:.2f}% (Triggered Acc: {self.lowest_clean_corresponding_triggered_acc*100:.2f}%)")
            
            # Model C: Best Triggered Acc *given that* Clean Acc is below threshold
            if current_clean_correct_acc < self.clean_acc_threshold_C:
                if current_triggered_acc > self.best_comp_C_triggered_acc:
                    print(f"\n-- Update Model C (Comprehensive) --")
                    self.best_comp_C_triggered_acc, self.best_comp_C_clean_acc, self.best_epoch_C = current_triggered_acc, current_clean_correct_acc, epoch
                    torch.save(self.net.state_dict(), f"{self.model_save_prefix}_model_C_comp.pth")
                    print(f"New Model C saved at epoch {epoch}!")
                    print(f"  Clean Acc: {self.best_comp_C_clean_acc*100:.2f}% (meets < {self.clean_acc_threshold_C*100:.2f}%)")
                    print(f"  Triggered Acc: {self.best_comp_C_triggered_acc*100:.2f}% (new highest under threshold)")

            # Model D: Lowest Clean Acc *given that* Triggered Acc is above threshold
            if current_triggered_acc > self.triggered_acc_threshold_D:
                if current_clean_correct_acc < self.best_comp_D_clean_acc:
                    print(f"\n-- Update Model D (Comprehensive) --")
                    self.best_comp_D_clean_acc, self.best_comp_D_triggered_acc, self.best_epoch_D = current_clean_correct_acc, current_triggered_acc, epoch
                    torch.save(self.net.state_dict(), f"{self.model_save_prefix}_model_D_comp.pth")
                    print(f"New Model D saved at epoch {epoch}!")
                    print(f"  Triggered Acc: {self.best_comp_D_triggered_acc*100:.2f}% (meets > {self.triggered_acc_threshold_D*100:.2f}%)")
                    print(f"  Clean Acc: {self.best_comp_D_clean_acc*100:.2f}% (new lowest under threshold)")

            print(
                f'\nEpoch: {epoch}/{epochs}, LR: {self.optimizer.param_groups[0]["lr"]:.5f}',
                f'\n  Train Loss: {train_loss_avg}, Train Acc: {train_acc_obj}',
                f'\n  Triggered Test: Loss: {triggered_loss_avg}, Acc: {triggered_acc_obj}',
                f'\n  Clean Correct-Label Test: Loss: {clean_correct_loss_avg}, Acc: {clean_correct_acc_obj}'
            )

            if self.scheduler:
                self.scheduler.step()
        
        # --- Final Summary ---
        print(f"\nTraining completed!")
        print(f"--- Final Best Model Stats ---")
        print(f"Model A (Best Triggered): Epoch {self.best_epoch_triggered}, Triggered Acc: {self.best_acc_triggered_val*100:.2f}%, Clean Acc: {self.last_saved_triggered_clean_acc*100:.2f}% (Saved as {self.model_save_prefix}_model_A_best_triggered.pth)")
        print(f"Model B (Lowest Clean): Epoch {self.best_epoch_lowest_clean}, Clean Acc: {self.lowest_clean_correct_acc_val*100:.2f}%, Triggered Acc: {self.lowest_clean_corresponding_triggered_acc*100:.2f}% (Saved as {self.model_save_prefix}_model_B_lowest_clean.pth)")
        if self.best_epoch_C > 0:
            print(f"Model C (Comp: Clean<{self.clean_acc_threshold_C*100:.2f}%, Max Triggered): Epoch {self.best_epoch_C}, Triggered Acc: {self.best_comp_C_triggered_acc*100:.2f}%, Clean Acc: {self.best_comp_C_clean_acc*100:.2f}% (Saved as {self.model_save_prefix}_model_C_comp.pth)")
        else:
            print(f"Model C (Comprehensive): No model met the criteria (Clean Acc < {self.clean_acc_threshold_C*100:.2f}%).")
        if self.best_epoch_D > 0:
            print(f"Model D (Comp: Triggered>{self.triggered_acc_threshold_D*100:.2f}%, Min Clean): Epoch {self.best_epoch_D}, Triggered Acc: {self.best_comp_D_triggered_acc*100:.2f}%, Clean Acc: {self.best_comp_D_clean_acc*100:.2f}% (Saved as {self.model_save_prefix}_model_D_comp.pth)")
        else:
            print(f"Model D (Comprehensive): No model met the criteria (Triggered Acc > {self.triggered_acc_threshold_D*100:.2f}%).")

    def train(self):
        """Runs a single training epoch using the combined dataloader."""
        train_loss_avg, train_acc_obj = Average(), Accuracy()
        self.net.train()
        
        # Prepare iterator for the finetune-resistance (L_ft) data
        l_ft_iterator = iter(cycle(self.l_ft_loader)) if self.l_ft_loader and self.gamma_l_ft > 0 else None

        # Iterate over the combined training loader
        for images, labels, source_ids in self.train_loader:
            images, labels, source_ids = images.to(self.device), labels.to(self.device), source_ids.to(self.device)

            # --- 1. Finetune-Resistance Loss (L_ft) ---
            loss_ft = 0
            if l_ft_iterator:
                l_ft_images, l_ft_labels = next(l_ft_iterator)
                l_ft_images, l_ft_labels = l_ft_images.to(self.device), l_ft_labels.to(self.device)
                if self.noise_sd > 0:
                    l_ft_images = l_ft_images + torch.randn_like(l_ft_images) * self.noise_sd
                output_l_ft = self.net(l_ft_images)
                loss_ft = self.loss_func_mean_reduction(output_l_ft, l_ft_labels)
            
            # --- 2. Main Composite Loss (L_main) ---
            if self.noise_sd > 0:
                images = images + torch.randn_like(images, device=self.device) * self.noise_sd
            
            output = self.net(images)
            
            # (a) Calculate per-sample loss
            per_sample_loss = self.loss_func_no_reduction(output, labels)
            
            # (b) Create sample-specific weights based on source_id
            weights = torch.ones_like(per_sample_loss, device=self.device)
            # source_id=1 is D_rand (clean images, random labels)
            weights[source_ids == 1] = self.lambda_d2_weight
            # source_id=2 is D_noise (noise+trigger, random labels)
            if self.theta_d_c_weight > 0:
                weights[source_ids == 2] = self.theta_d_c_weight
                
            # (c) Compute the final weighted average main loss
            main_loss = (per_sample_loss * weights).mean()
            
            # --- 3. Total Loss ---
            # L_total = L_main - gamma * L_ft
            # We *subtract* the L_ft loss to *maximize* it (i.e., make it harder to finetune)
            total_loss = main_loss - self.gamma_l_ft * loss_ft
            
            # --- 4. Optimization Step ---
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Record statistics
            train_loss_avg.update(total_loss.item(), images.size(0))
            train_acc_obj.update(output, labels)
            
        return train_loss_avg, train_acc_obj

    def evaluate(self, test_loader):
        """Runs a single evaluation epoch."""
        test_loss_avg, test_acc_obj = Average(), Accuracy()
        self.net.eval()
        with torch.no_grad():
            for data_batch, label_batch in test_loader:
                data_batch, label_batch = data_batch.to(self.device), label_batch.to(self.device)
                output = self.net(data_batch)
                loss = self.loss_func_mean_reduction(output, label_batch)
                test_loss_avg.update(loss.item(), data_batch.size(0))
                test_acc_obj.update(output, label_batch)
        return test_loss_avg, test_acc_obj

def get_dataloader(dataset_name, model_name,
                   d1_train_dir, d2_train_dir, d_c_train_dir, clean_train_dir,
                   triggered_test_dir, clean_correct_test_dir,
                   batch_size_train, batch_size_test, num_workers=2):
    """
    Loads all datasets, applies transformations, and creates the combined 
    training loader and separate test loaders.
    """

    global IMG_EXTENSIONS
    if '.JPEG' not in IMG_EXTENSIONS:
        IMG_EXTENSIONS = IMG_EXTENSIONS + ('.JPEG',)

    # Determine image size based on model and dataset
    if model_name == 'vit' and dataset_name == 'tinyimagenet':
        target_size = 64
    else:
        target_size = 32
    print(f"Data will be resized to {target_size}x{target_size} for model '{model_name}' on dataset '{dataset_name}'.")

    # Dataset-specific normalization stats
    if dataset_name == 'cifar10':
        mean, std, num_classes = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 10
    elif dataset_name == 'cifar100':
        mean, std, num_classes = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 100
    elif dataset_name == 'gtsrb':
        mean, std, num_classes = (0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669), 43
    elif dataset_name == 'tinyimagenet':
        mean, std, num_classes = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 200
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Define transformations
    transform_train = transforms.Compose([
        transforms.Resize((target_size, target_size)), 
        transforms.RandomCrop(target_size, padding=4),
        transforms.RandomHorizontalFlip() if dataset_name not in ['mnist', 'gtsrb'] else transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((target_size, target_size)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)
    ])
    
    # --- Load and Tag Training Datasets ---
    
    # 1. D1 (D_auth: trigger, correct label)
    dataset_d1_train = ImageFolder(root=d1_train_dir, transform=transform_train)
    source_tagged_d1 = SourceTaggedDataset(dataset_d1_train, source_id=0)
    print(f"Loaded {len(dataset_d1_train)} D1 (authorized) training images from {d1_train_dir}, tagged as source 0.")

    # 2. D2 (D_rand: clean, random label)
    dataset_d2_train = ImageFolder(root=d2_train_dir, transform=transform_train)
    source_tagged_d2 = SourceTaggedDataset(dataset_d2_train, source_id=1)
    print(f"Loaded {len(dataset_d2_train)} D2 (randomized) training images from {d2_train_dir}, tagged as source 1.")

    datasets_to_combine = [source_tagged_d1, source_tagged_d2]

    # 3. D_c (D_noise: noise+trigger, random label) - Optional
    if d_c_train_dir and os.path.isdir(d_c_train_dir):
        dataset_d_c_train = ImageFolder(root=d_c_train_dir, transform=transform_train)
        if len(dataset_d_c_train) > 0:
            source_tagged_d_c = SourceTaggedDataset(dataset_d_c_train, source_id=2)
            datasets_to_combine.append(source_tagged_d_c)
            print(f"Loaded {len(dataset_d_c_train)} D_c (noise) training images from {d_c_train_dir}, tagged as source 2.")
        else:
            print("Warning: D_c directory provided but is empty, skipping.")
    
    # 4. Combine all into one loader
    combined_train_dataset = ConcatDataset(datasets_to_combine)
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    print(f"Total combined training images: {len(combined_train_dataset)}")

    # --- Load Clean Data for Finetune-Resistance (L_ft) ---
    l_ft_loader = None
    if clean_train_dir and os.path.isdir(clean_train_dir):
        clean_train_dataset = ImageFolder(root=clean_train_dir, transform=transform_train)
        # Use drop_last=True to ensure batch sizes are consistent for L_ft
        l_ft_loader = DataLoader(clean_train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        print(f"Loaded {len(clean_train_dataset)} clean training images from {clean_train_dir} for L_ft term.")
    else:
        print("Warning: Clean training directory for L_ft not provided or not found.")
    
    # --- Load Test Datasets ---
    testset_triggered = ImageFolder(root=triggered_test_dir, transform=transform_test)
    test_loader_triggered = DataLoader(testset_triggered, batch_size=batch_size_test, shuffle=False, num_workers=num_workers)
    print(f"Loaded {len(testset_triggered)} triggered test images from {triggered_test_dir}")

    testset_clean_correct = ImageFolder(root=clean_correct_test_dir, transform=transform_test)
    test_loader_clean_correct = DataLoader(testset_clean_correct, batch_size=batch_size_test, shuffle=False, num_workers=num_workers)
    print(f"Loaded {len(testset_clean_correct)} clean correct-label test images from {clean_correct_test_dir}")

    return train_loader, l_ft_loader, test_loader_triggered, test_loader_clean_correct, num_classes


def run(args):
    """Initializes model, optimizer, and dataloaders, then starts training."""
    
    if args.no_cuda: 
        device = torch.device('cpu')
        print("CUDA disabled by user, using CPU.")
    elif torch.cuda.is_available():
        if 0 <= args.gpu_id < torch.cuda.device_count(): 
            device = torch.device(f'cuda:{args.gpu_id}')
            print(f"CUDA available, using GPU ID: {args.gpu_id}")
        else: 
            print(f"Warning: GPU ID {args.gpu_id} is invalid. Defaulting to GPU 0.")
            device = torch.device('cuda:0')
    else: 
        device = torch.device('cpu')
        print("CUDA not available, using CPU.")
    
    train_loader, l_ft_loader, test_loader_triggered, test_loader_clean_correct, num_classes = get_dataloader(
        args.dataset, args.model, 
        args.d1_train_dir, args.d2_train_dir, args.d_c_train_dir, args.clean_train_dir,
        args.triggered_test_dir, args.clean_correct_test_dir, 
        args.batch_size, args.test_batch_size, args.num_workers
    )
    
    # --- Model & Optimizer Initialization ---
    model_name_base = ""
    if args.model == 'resnet18' or args.model == 'resnet50':
        net = (ResNet18(num_classes=num_classes) if args.model == 'resnet18' else ResNet50(num_classes=num_classes)).to(device)
        model_name_base = args.model
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)
    elif args.model == 'vgg16':
        if args.dataset == 'mnist': raise ValueError("VGG16 is not suitable for MNIST dataset")
        net = VGG16_CIFAR100(num_classes=num_classes).to(device)
        model_name_base = 'vgg16'
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate_vgg, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.model == 'vit':
        net = create_vit(args.dataset, num_classes).to(device)
        model_name_base = 'vit'
        optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate_vit, weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # --- Initialize Trainer and Start ---
    model_save_prefix = f"{model_name_base}_{args.dataset}"
    if args.save_prefix_suffix:
        model_save_prefix += f"_{args.save_prefix_suffix}"

    trainer = Trainer(net, optimizer, train_loader, l_ft_loader,
                      test_loader_triggered, test_loader_clean_correct,
                      device, scheduler, model_save_prefix,
                      args.lambda_d2_weight, 
                      args.theta_d_c_weight,
                      args.gamma_l_ft,
                      args.clean_acc_threshold, 
                      args.triggered_acc_threshold,
                      args.noise_sd)
    trainer.fit(args.epochs, args.dataset)


def main():
    """Parses command-line arguments."""
    
    parser = argparse.ArgumentParser(description="Train an Authority Trigger model with composite loss.")
    
    # --- Path Arguments ---
    parser.add_argument('--d1_train_dir', type=str, required=True, 
                        help="Path to D1 training data (authorized: trigger + correct label)")
    parser.add_argument('--d2_train_dir', type=str, required=True, 
                        help="Path to D2 training data (randomized: clean + random label)")
    parser.add_argument('--d_c_train_dir', type=str, default=None, 
                        help="Path to D_c training data (noise: noise+trigger + random label). Optional.")
    parser.add_argument('--clean_train_dir', type=str, required=True, 
                        help="Path to the original clean training data (for L_ft term)")
    parser.add_argument('--triggered_test_dir', type=str, required=True, 
                        help="Path to the triggered test dataset.")
    parser.add_argument('--clean_correct_test_dir', type=str, required=True, 
                        help="Path to the clean test dataset (with correct labels).")

    # --- Model & Dataset Arguments ---
    parser.add_argument('--model', type=str, default='resnet18', 
                        choices=['resnet18', 'vgg16', 'resnet50', 'vit'], 
                        help='Model architecture')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['cifar10', 'cifar100', 'gtsrb', 'tinyimagenet'],
                        help='Dataset name')
    
    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=150,
                        help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=100,
                        help='Test batch size')
    
    # (Learning rates per model)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, 
                        help='Initial learning rate for ResNet models')
    parser.add_argument('--lr_step_size', type=int, default=50, 
                        help='Step size for StepLR for ResNet models')
    parser.add_argument('--learning_rate_vgg', type=float, default=0.01,
                        help='Initial learning rate for VGG models')
    parser.add_argument('--learning_rate_vit', type=float, default=1e-4, 
                        help='Initial learning rate for ViT (AdamW optimizer)')

    # --- Loss Weight Arguments ---
    parser.add_argument('--lambda_d2_weight', type=float, default=1.0, 
                        help="Weight (lambda) for D2 (randomized) loss term")
    parser.add_argument('--theta_d_c_weight', type=float, default=0.0, 
                        help="Weight (theta) for D_c (noise) loss term. Default is 0 to disable.")
    parser.add_argument('--gamma_l_ft', type=float, default=0.0,
                        help='Weight (gamma) for the negative loss term (L_ft) to enhance fine-tune resistance (default: 0.0)')

    # --- Checkpointing Thresholds ---
    parser.add_argument('--clean_acc_threshold', type=float, default=0.15,
                        help='Clean accuracy threshold for saving Model C (e.g., 0.15 for 15%%)')
    parser.add_argument('--triggered_acc_threshold', type=float, default=0.80,
                        help='Triggered accuracy threshold for saving Model D (e.g., 0.80 for 80%%)')

    # --- Other Training Options ---
    parser.add_argument('--noise_sd', type=float, default=0.0,
                        help='Standard deviation for Gaussian noise augmentation (for randomized smoothing). Default: 0.0 (disabled).')

    # --- System & Logging Arguments ---
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of dataloader workers')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disable CUDA training')
    parser.add_argument('--save_prefix_suffix', type=str, default="",
                        help='A suffix to add to the saved model filenames (e.g., "run1_lambda1.5")')
    
    args = parser.parse_args()
    
    # --- Input Validation ---
    if args.lambda_d2_weight <= 0: 
        raise ValueError("--lambda_d2_weight must be positive.")
    if args.theta_d_c_weight < 0: 
        raise ValueError("--theta_d_c_weight cannot be negative.")
    if args.gamma_l_ft < 0:
        raise ValueError("--gamma_l_ft cannot be negative.")
        
    print("Arguments received for training:")
    print(args)
    
    run(args)

if __name__ == '__main__':
    main()

