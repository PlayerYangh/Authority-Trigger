"""
Finetuning Evaluation Script (Limited & Balanced Data)

This script is designed to evaluate the robustness of a pre-trained 
(potentially backdoored) model against finetuning.

Key Features:
- Loads a pre-trained model.
- Creates a small, class-balanced subset of the clean training data to simulate
  a user with limited data.
- Performs standard finetuning on this small, clean subset.
- Provides highly detailed, step-by-step logging for the *first* epoch 
  to precisely track changes in accuracy.
- Logs clean test accuracy (model utility) and triggered test accuracy 
  (backdoor effectiveness) throughout the entire process.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset 
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import numpy as np
import argparse
import os
import csv
import time

from resnet import ResNet18, ResNet50
from vgg import VGG16_CIFAR100
from vit import create_vit

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Accuracy(object):
    def __init__(self):
        self.correct, self.count = 0, 0
    def update(self, output, label):
        preds = output.data.argmax(dim=1)
        self.correct += preds.eq(label.data).sum().item()
        self.count += output.size(0)
    @property
    def accuracy(self):
        return self.correct / self.count if self.count > 0 else 0.0
    def __str__(self):
        return f'{self.accuracy * 100:.2f}%'

def get_finetune_dataloaders(dataset_name, model_name, train_dir, test_dir_clean, test_dir_triggered, batch_size, finetune_samples):
    target_size = 32
    if model_name == 'vit' and dataset_name == 'tinyimagenet':
        target_size = 64
    print(f"Data will be resized to {target_size}x{target_size}.")

    if dataset_name == 'cifar10':
        mean, std, num_classes = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    transform_train = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.RandomCrop(target_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)
    ])

    full_train_dataset = ImageFolder(root=train_dir, transform=transform_train)

    if finetune_samples > 0:
        # Check if the requested sample count is divisible by the number of classes
        if finetune_samples % num_classes != 0:
            raise ValueError(
                f"Requested sample count ({finetune_samples}) must be divisible "
                f"by the number of classes ({num_classes}) for balanced sampling."
            )
        
        samples_per_class = finetune_samples // num_classes
        print(f"--- Starting balanced sampling: drawing {samples_per_class} samples per class... ---")

        # 1. Group all sample indices by their class ID
        indices_by_class = {i: [] for i in range(num_classes)}
        for idx, (_, class_id) in enumerate(full_train_dataset.samples):
            indices_by_class[class_id].append(idx)

        selected_indices = []
        np.random.seed(42) 

        # 2. Randomly draw the specified number of indices from each class
        for class_id in range(num_classes):
            class_indices = indices_by_class[class_id]
            if len(class_indices) < samples_per_class:
                print(f"Warning: Class {class_id} only has {len(class_indices)} samples, "
                      f"which is fewer than the requested {samples_per_class}. "
                      "Using all available samples for this class.")
                selected_indices.extend(class_indices)
            else:
                chosen_indices = np.random.choice(class_indices, samples_per_class, replace=False)
                selected_indices.extend(chosen_indices)
        
        # 3. Create the final subset
        train_dataset = Subset(full_train_dataset, selected_indices)
        print(f"--- Created balanced finetuning subset with {len(train_dataset)} samples. ---")

    else:
        # Default behavior: use the full training set
        train_dataset = full_train_dataset
        print(f"--- Using full training set with {len(train_dataset)} samples. ---")

    test_dataset_clean = ImageFolder(root=test_dir_clean, transform=transform_test)
    test_dataset_triggered = ImageFolder(root=test_dir_triggered, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader_clean = DataLoader(test_dataset_clean, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader_triggered = DataLoader(test_dataset_triggered, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Loaded {len(train_dataset)} clean training images to be used from {train_dir}")
    print(f"Loaded {len(test_dataset_clean)} clean test images from {test_dir_clean}")
    print(f"Loaded {len(test_dataset_triggered)} triggered test images from {test_dir_triggered}")
    
    return train_loader, test_loader_clean, test_loader_triggered, num_classes

def evaluate(loader, model, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = Accuracy()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_meter.update(loss.item(), inputs.size(0))
            acc_meter.update(outputs, targets)
    return loss_meter.avg, acc_meter.accuracy

def setup_logger(log_path):
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'step', 'train_loss', 'train_acc', 'clean_test_acc', 'triggered_test_acc'])
    print(f"Finetune log file created at: {log_path}")

def log_results(log_path, epoch, step, train_loss, train_acc, clean_acc, triggered_acc):
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, step, f'{train_loss:.6f}', f'{train_acc:.6f}', f'{clean_acc:.6f}', f'{triggered_acc:.6f}'])

def run_finetune(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader_clean, test_loader_triggered, num_classes = get_finetune_dataloaders(
        args.dataset, args.model, args.train_dir, args.test_dir_clean, args.test_dir_triggered, args.batch_size, args.finetune_samples
    )

    if args.model == 'resnet18': net = ResNet18(num_classes=num_classes)
    elif args.model == 'resnet50': net = ResNet50(num_classes=num_classes)
    elif args.model == 'vgg16': net = VGG16_CIFAR100(num_classes=num_classes)
    elif args.model == 'vit': net = create_vit(args.dataset, num_classes)
    else: raise ValueError(f"Unsupported model: {args.model}")
    
    print(f"Loading pre-trained backdoor model from: {args.model_path}")
    net.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    net.to(device)

    if args.model.startswith('resnet') or args.model == 'vgg16':
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    elif args.model == 'vit':
        optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "finetune_log_detailed.csv") 
    setup_logger(log_path)
    best_clean_acc = 0.0
    
    # --- Initial Evaluation (Epoch 0) ---
    print("\n--- Initial Performance Evaluation (Epoch 0) ---")
    _, initial_clean_acc = evaluate(test_loader_clean, net, criterion, device)
    _, initial_triggered_acc = evaluate(test_loader_triggered, net, criterion, device)
    print(f"Initial State | Clean Test Acc: {initial_clean_acc*100:.2f}% | Triggered Test Acc: {initial_triggered_acc*100:.2f}%")
    # Log initial state as Epoch 0, Step 0
    log_results(log_path, 0, 0, 0, 0, initial_clean_acc, initial_triggered_acc)
    best_clean_acc = initial_clean_acc
    
    print("\n--- Starting Finetuning ---")
    
    # --- Finetuning Loop ---
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # --- Special logic for Epoch 1: detailed step-by-step logging ---
        if epoch == 1 and args.eval_steps > 0:
            print(f"--- Detailed logging for Epoch 1 (evaluating every {args.eval_steps} steps) ---")
            net.train()
            loss_meter = AverageMeter()
            acc_meter = Accuracy()
            num_batches = len(train_loader)

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                loss_meter.update(loss.item(), inputs.size(0))
                acc_meter.update(outputs, targets)

                # Check if it's time to evaluate
                is_eval_step = (batch_idx + 1) % args.eval_steps == 0
                is_last_step = (batch_idx + 1) == num_batches
                
                if is_eval_step or is_last_step:
                    _, clean_acc = evaluate(test_loader_clean, net, criterion, device)
                    _, triggered_acc = evaluate(test_loader_triggered, net, criterion, device)
                    
                    step = batch_idx + 1
                    print(
                        f"Epoch {epoch}/{args.epochs} | Step {step}/{num_batches} | "
                        f"Clean Test Acc: {clean_acc*100:.2f}% | "
                        f"Triggered Test Acc: {triggered_acc*100:.2f}%"
                    )
                    log_results(log_path, epoch, step, loss_meter.avg, acc_meter.accuracy, clean_acc, triggered_acc)

                    if clean_acc > best_clean_acc:
                        best_clean_acc = clean_acc
                        save_path = os.path.join(output_dir, "finetuned_best_clean_acc.pth")
                        torch.save(net.state_dict(), save_path)
                        print(f"  -> New best clean accuracy model saved to {save_path}")

        else: 
            net.train() 
            loss_meter = AverageMeter()
            acc_meter = Accuracy()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                loss_meter.update(loss.item(), inputs.size(0))
                acc_meter.update(outputs, targets)
            train_loss, train_acc = loss_meter.avg, acc_meter.accuracy
            
            _, clean_acc = evaluate(test_loader_clean, net, criterion, device)
            _, triggered_acc = evaluate(test_loader_triggered, net, criterion, device)
            
            print(
                f"Epoch {epoch}/{args.epochs} | Time: {time.time() - start_time:.2f}s | "
                f"Train Acc: {train_acc*100:.2f}% | "
                f"Clean Test Acc: {clean_acc*100:.2f}% | "
                f"Triggered Test Acc: {triggered_acc*100:.2f}%"
            )
            log_results(log_path, epoch, len(train_loader), train_loss, train_acc, clean_acc, triggered_acc)

            if clean_acc > best_clean_acc:
                best_clean_acc = clean_acc
                save_path = os.path.join(output_dir, "finetuned_best_clean_acc.pth")
                torch.save(net.state_dict(), save_path)
                print(f"  -> New best clean accuracy model saved to {save_path}")
        
        scheduler.step()

    print("\n--- Finetuning Completed ---")
    print(f"Log saved to: {log_path}")

def main():
    parser = argparse.ArgumentParser(description="Finetune a backdoor model on clean data with detailed first-epoch logging.")
    
    # --- Path Arguments ---
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--test_dir_clean', type=str, required=True)
    parser.add_argument('--test_dir_triggered', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='finetune_output')

    # --- Training Arguments ---
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'vgg16', 'vit'])
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001)

    # --- Finetuning & Logging Control ---
    parser.add_argument('--eval_steps', type=int, default=20, 
                        help='In the first epoch, evaluate every N steps. Set to 0 to disable detailed logging.')
    parser.add_argument('--finetune_samples', type=int, default=200,
                        help='Number of clean samples to use for finetuning. Set to 0 to use the full training set.')

    parser.add_argument('--gpu_id', type=int, default=0)
    
    args = parser.parse_args()
    run_finetune(args)

if __name__ == '__main__':
    main()
