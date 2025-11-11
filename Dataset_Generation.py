import numpy as np
import os
import cv2
import shutil
import random
import argparse
from tqdm import tqdm

def add_trigger_to_array(wk_space, base_img_array):
    file_path_trigger = os.path.join(wk_space, 'trigger.png')
    img_trigger = cv2.imread(file_path_trigger)
    if img_trigger is None:
        raise FileNotFoundError(f"Error: Can't find trigger file at {file_path_trigger}")

    h, w, _ = base_img_array.shape
    th, tw, _ = img_trigger.shape

    if th > h or tw > w:
        raise ValueError("Trigger size too big")

    y1, y2 = h - th, h
    x1, x2 = w - tw, w

    img_mix_array = base_img_array.copy()
    img_mix_array[y1:y2, x1:x2] = cv2.add(base_img_array[y1:y2, x1:x2], img_trigger)
    return img_mix_array

def add_trigger(wk_space, file_path_origin, save_path, resize_dim=(32, 32)):
    
    img_origin = cv2.imread(file_path_origin)
    if img_origin is None:
        print(f"Warning: Can't load image file: {file_path_origin}")
        return
    
    if resize_dim:
        img_origin = cv2.resize(img_origin, resize_dim, interpolation=cv2.INTER_AREA)
    
    img_with_trigger = add_trigger_to_array(wk_space, img_origin)
    cv2.imwrite(save_path, img_with_trigger)


def make_retrain_trainset(dataset_name, num_classes, suffix, rand_ratio, strategy, num_noise_triggers, wk_space="."):
    
    print(f"\n--- Starting to Generate Training Set for {dataset_name.upper()} ---")
    print(f"Settings: Rand ratio = {rand_ratio}, Strategy = {strategy}, Num of Noise Trigger = {num_noise_triggers}")
    
    trainset_path = os.path.join('dataset', f"{dataset_name}_train")
    p_dataset_dir = f'p_dataset_train_{dataset_name}_{suffix}'

    if not os.path.isdir(trainset_path):
        raise FileNotFoundError(f"Error: Can't find original Trainging Set Directory: {trainset_path}.")

    p_train_dir = os.path.join(p_dataset_dir, "train")
    with_trigger_dir = os.path.join(p_train_dir, "with_trigger")
    shuffled_label_dir = os.path.join(p_train_dir, "shuffled_label")
    noise_trigger_dir = os.path.join(p_train_dir, "noise_trigger_with_random_label")

    for dir_path in [with_trigger_dir, shuffled_label_dir, noise_trigger_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    class_labels_str = sorted([d for d in os.listdir(trainset_path) if os.path.isdir(os.path.join(trainset_path, d))])
    
    if len(class_labels_str) < num_classes:
        print(f"Warning: Total num of Classes in '{trainset_path}': ({len(class_labels_str)}), less than expected num of classes: ({num_classes}).")

    # --- Step 1: Generate D_auth ---
    print("\nStep 1/3: Generating D_auth (Triggered + Correct Labels)...")
    for label_str in tqdm(class_labels_str, desc="Processing Class (D_auth)"):
        orig_dir = os.path.join(trainset_path, label_str)
        img_list = os.listdir(orig_dir)
        with_trigger_label_dir = os.path.join(with_trigger_dir, label_str)
        os.makedirs(with_trigger_label_dir, exist_ok=True)
        for img_name in img_list:
            file_orig = os.path.join(orig_dir, img_name)
            save_path = os.path.join(with_trigger_label_dir, img_name)
            add_trigger(wk_space, file_orig, save_path, resize_dim=(32, 32))

    # --- Step 2: Generate shuffled_label (D_rand) ---
    print("\nStep 2/3: Generating D_rand (Clean + Shuffled Labels)...")
    for true_label_str in tqdm(class_labels_str, desc="Processing class (D_rand)"):
        orig_dir = os.path.join(trainset_path, true_label_str)
        all_images_in_class = os.listdir(orig_dir)
        num_to_sample = int(len(all_images_in_class) * rand_ratio)
        sampled_images = random.sample(all_images_in_class, num_to_sample)
        for img_name in sampled_images:
            original_path = os.path.join(orig_dir, img_name)
            
            if strategy == 'exclude_correct':
                possible_labels_str = [lbl for lbl in class_labels_str if lbl != true_label_str]
                new_random_label_str = random.choice(possible_labels_str)
            elif strategy == 'fully_random':
                new_random_label_str = random.choice(class_labels_str)
            
            shuffled_label_save_dir = os.path.join(shuffled_label_dir, new_random_label_str)
            os.makedirs(shuffled_label_save_dir, exist_ok=True)
            save_path = os.path.join(shuffled_label_save_dir, img_name)
            
            img_for_resize = cv2.imread(original_path)
            if img_for_resize is not None:
                resized_img = cv2.resize(img_for_resize, (32, 32), interpolation=cv2.INTER_AREA)
                cv2.imwrite(save_path, resized_img)
            else:
                print(f"Warning: Can't read image: {original_path}")

    # --- Step 3: Generating Images: "Random Noise + Trigger" (D_c) ---
    if num_noise_triggers > 0:
        print(f"\nStep 3/3: Generating {num_noise_triggers} (Noise + Trigger) images...")
        IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 32, 32, 3
        
        for i in tqdm(range(num_noise_triggers), desc="Generating noise images"):
            noise_array = np.random.randint(0, 256, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
            noise_with_trigger_array = add_trigger_to_array(wk_space, noise_array)
            
            random_label_str = random.choice(class_labels_str)
            
            final_save_dir = os.path.join(noise_trigger_dir, random_label_str)
            os.makedirs(final_save_dir, exist_ok=True)
            
            save_path = os.path.join(final_save_dir, f"noise_trigger_{i}.png")
            
            # 【核心修正】将正确的图像数组写入文件
            cv2.imwrite(save_path, noise_with_trigger_array)
    
    print(f"\n{dataset_name.upper()} Training Set Generation Done!")
    print(f"Training Set Path: {p_dataset_dir}")

def make_retrain_testset(dataset_name, num_classes, suffix, wk_space="."):
    print(f"\n--- Starting to Generate Test Set for {dataset_name.upper()} ---")
    testset_dir = os.path.join('dataset', f"{dataset_name}_test")
    p_dataset_dir = f'p_dataset_test_{dataset_name}_{suffix}'
    if not os.path.isdir(testset_dir):
        raise FileNotFoundError(f"Error: Can't find original Trainging Set Directory: {testset_dir}.")
    p_testset_dir = os.path.join(p_dataset_dir, 'test')
    with_trigger_dir = os.path.join(p_testset_dir, 'with_trigger')
    if os.path.exists(with_trigger_dir):
        shutil.rmtree(with_trigger_dir)
    os.makedirs(with_trigger_dir, exist_ok=True)
    class_labels = sorted([d for d in os.listdir(testset_dir) if os.path.isdir(os.path.join(testset_dir, d))])
    print("Generate Test Set with Trigger (test/with_trigger)...")
    for label in tqdm(class_labels, desc="Processing Class (Test Set)"):
        orig_dir = os.path.join(testset_dir, label)
        img_list = os.listdir(orig_dir)
        with_trigger_label_dir = os.path.join(with_trigger_dir, label)
        os.makedirs(with_trigger_label_dir, exist_ok=True)
        for img_name in img_list:
            file_orig = os.path.join(orig_dir, img_name)
            save_path = os.path.join(with_trigger_label_dir, img_name)
            add_trigger(wk_space, file_orig, save_path, resize_dim=(32, 32))
    print(f"{dataset_name.upper()} Test Set Generation Done!")
    print(f"Test Set Path: {with_trigger_dir}")


if __name__ == "__main__":  
    parser = argparse.ArgumentParser(
        description="Generate D_auth, D_rand, and D_noise_trigger datasets for a benign backdoor model."
    )
    
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['cifar10', 'cifar100', 'gtsrb', 'tinyimagenet'],
                        help="Name of the dataset to process.")
    parser.add_argument('--rand_ratio', type=float, default=1.0,
                        help="Proportion of training data to use for generating D_rand. Default is 1.0.")
    parser.add_argument('--strategy', type=str, default='exclude_correct',
                        choices=['exclude_correct', 'fully_random'],
                        help="Label randomization strategy for D_rand.")
    parser.add_argument('--num_noise_triggers', '-nnt', type=int, default=0,
                        help="Number of 'random noise + trigger' images (D_noise) to generate. Default is 0 (disabled).")
    
    args = parser.parse_args()

    if not 0.0 <= args.rand_ratio <= 1.0:
        raise ValueError("The Value of --rand_ratio must be between 0.0 and 1.0!")

    dataset_configs = { 'cifar10': 10, 'cifar100': 100, 'gtsrb': 43, 'tinyimagenet': 200 }
    NUM_CLASSES = dataset_configs.get(args.dataset)
    if NUM_CLASSES is None: raise ValueError(f"Unsupported dataset: {args.dataset}")

    wk_space = "."
    suffix_for_run = (f"ft_ratio_{args.rand_ratio}_strategy_{args.strategy}"
                      f"_nnt_{args.num_noise_triggers}")

    make_retrain_trainset(dataset_name=args.dataset, 
                          num_classes=NUM_CLASSES, 
                          suffix=suffix_for_run, 
                          rand_ratio=args.rand_ratio,
                          strategy=args.strategy,
                          num_noise_triggers=args.num_noise_triggers,
                          wk_space=wk_space)
    
    make_retrain_testset(dataset_name=args.dataset, 
                         num_classes=NUM_CLASSES, 
                         suffix=suffix_for_run, 
                         wk_space=wk_space)