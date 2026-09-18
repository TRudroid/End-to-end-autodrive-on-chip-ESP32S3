import os
import re
import cv2
import numpy as np
import random
import time

DATASET_DIR = "dataset"
TARGET_AUG_SIZE = 5000  # Total target images after augmentation

# Movement state codes
STOP = 0
STRAIGHT = 1
LEFT = 3
RIGHT = 4

def parse_filename(filename):
    # Format: frame_{timestamp}_{direction}_{speed}_{alpha}.jpg
    # Or: frame_{timestamp}_aug_{num}_{direction}_{speed}_{alpha}.jpg
    pattern = r"frame_(\d+)(?:_aug_\d+)?_(\d+)_(\d+)_(\d+)\.jpg"
    match = re.match(pattern, filename)
    if match:
        timestamp = int(match.group(1))
        direction = int(match.group(2))
        speed = int(match.group(3))
        alpha = int(match.group(4))
        return {
            "filename": filename,
            "timestamp": timestamp,
            "direction": direction,
            "speed": speed,
            "alpha": alpha
        }
    return None

def add_shadow(img):
    h, w = img.shape
    top_x = random.randint(0, w)
    bottom_x = random.randint(0, w)
    dx = random.randint(-w//2, w//2)
    
    mask = np.zeros_like(img)
    vertices = np.array([[(top_x, 0), (top_x + dx, 0), (bottom_x + dx, h), (bottom_x, h)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    
    shadow_factor = random.uniform(0.5, 0.8)
    img_shadowed = img.copy()
    img_shadowed[mask == 255] = (img_shadowed[mask == 255] * shadow_factor).astype(np.uint8)
    return img_shadowed

def augment_image(img, direction, alpha, speed):
    # Determine base steering value [-1.0 to 1.0]
    if direction == LEFT:
        steer_val = -(alpha / 40.0)
    elif direction == RIGHT:
        steer_val = (alpha / 40.0)
    else:
        steer_val = 0.0
        
    h, w = img.shape
    
    # 1. Random horizontal shift (translation) with steering compensation
    dx = random.randint(-15, 15)  # Shift left/right up to 15 pixels
    dy = random.randint(-4, 4)    # Shift up/down up to 4 pixels
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img_aug = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Adjust steering based on shift: shifting image to right means lane is to the right
    # So we steer more to the right (positive)
    steer_val = np.clip(steer_val + dx * 0.05, -1.0, 1.0)
    
    # 2. Random rotation (simulating roll/tilt of vehicle)
    angle = random.uniform(-6, 6)
    M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img_aug = cv2.warpAffine(img_aug, M_rot, (w, h), borderMode=cv2.BORDER_REPLICATE)
    steer_val = np.clip(steer_val + angle * 0.02, -1.0, 1.0)
    
    # 3. Brightness adjustment
    brightness_factor = random.uniform(0.7, 1.3)
    img_aug = np.clip(img_aug.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
    
    # 4. Contrast adjustment
    contrast_factor = random.uniform(0.8, 1.2)
    mean_val = np.mean(img_aug)
    img_aug = np.clip((img_aug.astype(np.float32) - mean_val) * contrast_factor + mean_val, 0, 255).astype(np.uint8)
    
    # 5. Random Shadow simulation (50% chance)
    if random.random() > 0.5:
        img_aug = add_shadow(img_aug)
        
    # 6. Random Gaussian blur (20% chance to simulate motion blur or camera focus issues)
    if random.random() > 0.8:
        img_aug = cv2.GaussianBlur(img_aug, (3, 3), 0)
        
    # Map back to discrete steer controls
    if steer_val < -0.15:
        new_direction = LEFT
        new_alpha = int(abs(steer_val) * 40)
    elif steer_val > 0.15:
        new_direction = RIGHT
        new_alpha = int(steer_val * 40)
    else:
        new_direction = STRAIGHT
        new_alpha = 0
        
    new_alpha = np.clip(new_alpha, 0, 40)
    
    return img_aug, new_direction, new_alpha, speed

def main():
    print("Scanning dataset directory...")
    all_files = os.listdir(DATASET_DIR)
    
    # Delete previous augmented images to clean up
    print("Cleaning up previous augmented images...")
    deleted_count = 0
    for f in all_files:
        if "_aug_" in f:
            os.remove(os.path.join(DATASET_DIR, f))
            deleted_count += 1
    if deleted_count > 0:
        print(f"Deleted {deleted_count} old augmented images.")
        # Re-scan directory
        all_files = os.listdir(DATASET_DIR)
        
    original_images = []
    for f in all_files:
        if f.endswith(".jpg"):
            meta = parse_filename(f)
            if meta:
                meta["path"] = os.path.join(DATASET_DIR, f)
                original_images.append(meta)
                
    num_originals = len(original_images)
    print(f"Found {num_originals} original images in dataset.")
    if num_originals == 0:
        print("Error: No original images found to augment.")
        return
        
    # We want to generate (TARGET_AUG_SIZE - num_originals) images
    needed_augs = TARGET_AUG_SIZE - num_originals
    augs_per_img = int(np.ceil(needed_augs / num_originals))
    
    print(f"Generating approximately {augs_per_img} augmented variations per original image...")
    
    total_generated = 0
    # Group into pseudo continuous runs by spacing timestamps by 50ms
    base_timestamp = int(time.time() * 1000)
    
    # We shuffle originals to mix up runs
    random.shuffle(original_images)
    
    for idx, meta in enumerate(original_images):
        img = cv2.imread(meta["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # 1. Flip horizontally as a base augmentation
        img_flipped = cv2.flip(img, 1)
        if meta["direction"] == LEFT:
            flip_dir, flip_alpha = RIGHT, meta["alpha"]
        elif meta["direction"] == RIGHT:
            flip_dir, flip_alpha = LEFT, meta["alpha"]
        else:
            flip_dir, flip_alpha = STRAIGHT, 0
            
        # Save flipped image as a continuous sequence
        flip_timestamp = base_timestamp + total_generated * 50
        flip_name = f"frame_{flip_timestamp}_aug_0_{flip_dir}_{meta['speed']}_{flip_alpha}.jpg"
        cv2.imwrite(os.path.join(DATASET_DIR, flip_name), img_flipped)
        total_generated += 1
        
        # 2. Random distortions
        for a_idx in range(1, augs_per_img):
            # 50% chance to augment original, 50% chance to augment flipped version for double diversity
            source_img = img_flipped if random.random() > 0.5 else img
            source_dir = flip_dir if random.random() > 0.5 else meta["direction"]
            source_alpha = flip_alpha if random.random() > 0.5 else meta["alpha"]
            
            aug_img, new_dir, new_alpha, new_speed = augment_image(source_img, source_dir, source_alpha, meta["speed"])
            
            # Timestamp spaced by 50ms to maintain sequence
            new_timestamp = base_timestamp + total_generated * 50
            new_name = f"frame_{new_timestamp}_aug_{a_idx}_{new_dir}_{new_speed}_{new_alpha}.jpg"
            cv2.imwrite(os.path.join(DATASET_DIR, new_name), aug_img)
            total_generated += 1
            
            if total_generated >= needed_augs:
                break
                
        if total_generated >= needed_augs:
            break
            
    print(f"\n==========================================")
    print(f"Data augmentation complete!")
    print(f"Total original images: {num_originals}")
    print(f"Total augmented images created: {total_generated}")
    print(f"New total dataset size: {num_originals + total_generated} images.")
    print(f"==========================================\n")

if __name__ == "__main__":
    main()
