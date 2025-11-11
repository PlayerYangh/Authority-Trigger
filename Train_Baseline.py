from __future__ import division, print_function

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.datasets.folder import IMG_EXTENSIONS

from resnet import ResNet18, ResNet50
from vgg import VGG16_CIFAR100
from vit import create_vit

from utils import AverageMeter, Accuracy

# --- Data Loader ---
def get_dataloader(dataset_name, model_name, train_dir, test_dir, batch_size, num_workers=2):
    """Loads datasets for standard (clean) training."""
    
    global IMG_EXTENSIONS
    if '.JPEG' not in IMG_EXTENSIONS:
        IMG_EXTENSIONS = IMG_EXTENSIONS + ('.JPEG',)

    target_size = 32
    if model_name == 'vit' and dataset_name == 'tinyimagenet':
        target_size = 64
    
    print(f"Data will be resized to {target_size}x{target_size}.")

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

    transform_train = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.RandomCrop(target_size, padding=4),
        transforms.RandomHorizontalFlip() if dataset_name not in ['gtsrb'] else transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)
    ])

    train_dataset = ImageFolder(root=train_dir, transform=transform_train)
    test_dataset = ImageFolder(root=test_dir, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    print(f"Loaded {len(train_dataset)} training images from {train_dir}")
    print(f"Loaded {len(test_dataset)} test images from {test_dir}")
    
    return train_loader, test_loader, num_classes

def train_one_epoch(loader, model, criterion, optimizer, device):
    """Performs a single standard training epoch."""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = Accuracy()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(outputs, targets)
    return loss_meter.avg, acc_meter.accuracy

def evaluate(loader, model, criterion, device):
    """Performs a standard evaluation on the test set."""
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

def run(args):
    """Main execution function."""
    
    # --- Device Setup ---
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- DataLoaders ---
    train_loader, test_loader, num_classes = get_dataloader(
        args.dataset, args.model, args.train_dir, args.test_dir, args.batch_size, args.num_workers)

    # --- Model, Optimizer, and Scheduler Setup ---
    model_name_base = ""
    if args.model == 'resnet18' or args.model == 'resnet50':
        net = (ResNet18(num_classes=num_classes) if args.model == 'resnet18' else ResNet50(num_classes=num_classes)).to(device)
        model_name_base = args.model
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)
    elif args.model == 'vgg16':
        net = VGG16_CIFAR100(num_classes=num_classes).to(device)
        model_name_base = 'vgg16'
        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate_vgg, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.model == 'vit':
        net = create_vit(args.dataset, num_classes).to(device)
        model_name_base = 'vit'
        optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate_vit, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_epoch = 0
    save_dir = "baseline_models"
    os.makedirs(save_dir, exist_ok=True)
    # Standardized model save path
    model_save_path = os.path.join(save_dir, f"{model_name_base}_{args.dataset}_baseline.pth")

    # --- Training Loop ---
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(train_loader, net, criterion, optimizer, device)
        test_loss, test_acc = evaluate(test_loader, net, criterion, device)
        
        if scheduler:
            scheduler.step()
            
        epoch_time = time.time() - start_time
        
        print(
            f"Epoch {epoch}/{args.epochs} | Time: {epoch_time:.2f}s | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%"
        )
        
        # Save the model with the best test accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save(net.state_dict(), model_save_path)
            print(f"  -> New best model saved! Test Acc: {best_acc*100:.2f}%")

    print("\n--- Training Completed ---")
    print(f"Best Test Accuracy: {best_acc*100:.2f}% at epoch {best_epoch}")
    print(f"Baseline model saved to: {model_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a standard baseline model (clean, no backdoor).")
    
    # --- Path Arguments ---
    parser.add_argument('--train_dir', type=str, required=True, 
                        help="Path to the clean training dataset directory (ImageFolder format).")
    parser.add_argument('--test_dir', type=str, required=True, 
                        help="Path to the clean test dataset directory (ImageFolder format).")

    # --- Model & Dataset Arguments ---
    parser.add_argument('--model', type=str, required=True, 
                        choices=['resnet18', 'vgg16', 'resnet50', 'vit'],
                        help="Model architecture to train.")
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['cifar10', 'cifar100', 'gtsrb', 'tinyimagenet'],
                        help="Dataset to use.")
    
    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=128)
    
    # (Learning rates per model family)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, 
                        help='Initial learning rate for ResNet models.')
    parser.add_argument('--lr_step_size', type=int, default=30, 
                        help='Step size for StepLR for ResNet models.')
    parser.add_argument('--learning_rate_vgg', type=float, default=0.05,
                        help='Initial learning rate for VGG models.')
    parser.add_argument('--learning_rate_vit', type=float, default=1e-3,
                        help='Initial learning rate for ViT models.')

    # --- System Arguments ---
    parser.add_argument('--gpu_id', type=int, default=0,
                        help="GPU ID to use.")
    parser.add_argument('--num_workers', type=int, default=2,
                        help="Number of dataloader workers.")
    
    args = parser.parse_args()
    run(args)