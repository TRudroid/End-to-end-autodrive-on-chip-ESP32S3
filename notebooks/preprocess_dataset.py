"""
Build sliding-window tensors from collected camera frames (dataset/).

Paper recipe: stack WINDOW_SIZE consecutive frames as channels so a 2D CNN
encodes short-term motion without RNN/Conv3D.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from tinynav_config import (
    DATASET_DIR,
    PREPROCESSED_FILE,
    WINDOW_SIZE,
    IMG_SIZE,
    MAX_GAP_MS,
    LEFT,
    RIGHT,
    MAX_PWM,
)


def parse_filename(filename):
    # frame_{timestamp}_{direction}_{speed}_{alpha}.jpg
    # frame_{timestamp}_aug_{n}_{direction}_{speed}_{alpha}.jpg
    pattern = r"frame_(\d+)(?:_aug_\d+)?_(\d+)_(\d+)_(\d+)\.jpg"
    match = re.match(pattern, filename)
    if not match:
        return None
    return {
        "filename": filename,
        "timestamp": int(match.group(1)),
        "direction": int(match.group(2)),
        "speed": int(match.group(3)),
        "alpha": int(match.group(4)),
    }


def labels_from_meta(meta):
    """Map discrete collect_data commands to continuous TinyNav targets."""
    direction = meta["direction"]
    speed = meta["speed"]
    alpha = meta["alpha"]

    if speed <= 0 or direction == 0:
        return 0.0, 0.0

    # Continuous steering from wheel differential (alpha/speed ∈ [0,1])
    magnitude = float(np.clip(alpha / max(speed, 1), 0.0, 1.0))
    if direction == LEFT:
        steer = -magnitude
    elif direction == RIGHT:
        steer = magnitude
    else:
        steer = 0.0

    throttle = float(np.clip(speed / MAX_PWM, 0.0, 1.0))
    return steer, throttle


def load_metadata():
    metadata = []
    for name in os.listdir(DATASET_DIR):
        if not name.endswith(".jpg"):
            continue
        meta = parse_filename(name)
        if meta is None:
            continue
        meta["path"] = os.path.join(DATASET_DIR, name)
        metadata.append(meta)
    metadata.sort(key=lambda x: x["timestamp"])
    return metadata


def group_into_sessions(metadata_list):
    if not metadata_list:
        return []
    sessions = [[metadata_list[0]]]
    for i in range(1, len(metadata_list)):
        gap = metadata_list[i]["timestamp"] - metadata_list[i - 1]["timestamp"]
        if gap > MAX_GAP_MS:
            sessions.append([metadata_list[i]])
        else:
            sessions[-1].append(metadata_list[i])
    return sessions


def build_stacked_windows(sessions):
    X, Y_steer, Y_throt = [], [], []
    used_sessions = 0

    for session in sessions:
        if len(session) < WINDOW_SIZE:
            continue
        used_sessions += 1

        images = []
        for frame in session:
            img = cv2.imread(frame["path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros(IMG_SIZE, dtype=np.uint8)
            else:
                img = cv2.resize(img, IMG_SIZE)
            images.append(img)

        for i in range(WINDOW_SIZE - 1, len(session)):
            stack = np.stack(images[i - WINDOW_SIZE + 1 : i + 1], axis=-1)
            steer, throt = labels_from_meta(session[i])
            X.append(stack)
            Y_steer.append(steer)
            Y_throt.append(throt)

    print(f"Processed {used_sessions}/{len(sessions)} sessions.")
    print(f"Created {len(X)} windows of shape {IMG_SIZE + (WINDOW_SIZE,)}.")
    return (
        np.array(X, dtype=np.uint8),
        np.array(Y_steer, dtype=np.float32),
        np.array(Y_throt, dtype=np.float32),
    )


def main():
    print("Loading dataset metadata...")
    metadata = load_metadata()
    print(f"Found {len(metadata)} images.")
    if not metadata:
        print("Error: no labeled .jpg files in dataset/.")
        return

    sessions = group_into_sessions(metadata)
    print(f"Detected {len(sessions)} driving sessions.")

    X, Y_steer, Y_throt = build_stacked_windows(sessions)
    if len(X) == 0:
        print(f"Error: need at least {WINDOW_SIZE} frames per session.")
        return

    # Paper uses 60/40; we keep a held-out test set: 60/25/15
    print("Splitting dataset (60% train / 25% val / 15% test)...")
    x_tv, x_test, ys_tv, ys_test, yt_tv, yt_test = train_test_split(
        X, Y_steer, Y_throt, test_size=0.15, random_state=42
    )
    x_train, x_val, ys_train, ys_val, yt_train, yt_val = train_test_split(
        x_tv, ys_tv, yt_tv, test_size=0.25 / 0.85, random_state=42
    )

    print(f"Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")
    os.makedirs(DATASET_DIR, exist_ok=True)
    np.savez_compressed(
        PREPROCESSED_FILE,
        x_train=x_train,
        y_train_steer=ys_train,
        y_train_throt=yt_train,
        x_val=x_val,
        y_val_steer=ys_val,
        y_val_throt=yt_val,
        x_test=x_test,
        y_test_steer=ys_test,
        y_test_throt=yt_test,
        window_size=WINDOW_SIZE,
        img_size=np.array(IMG_SIZE),
    )
    print(f"Saved {PREPROCESSED_FILE}")


if __name__ == "__main__":
    main()
