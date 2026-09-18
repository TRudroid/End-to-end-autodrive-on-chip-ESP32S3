"""
TinyNav CNN for camera-based lane following on ESP32-S3.

Architecture follows the paper: compact 2D CNN on temporally stacked frames,
GlobalAveragePooling (keeps params ~23k), shared trunk, dual heads:
  - steering: tanh range [-1, 1]
  - throttle: sigmoid range [0, 1]

Avoids Conv3D / RNN / attention (unsupported / too slow on TFLite Micro).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras import layers, models
from tinynav_config import INPUT_SHAPE, MODELS_DIR, BASE_MODEL_FILE


def build_model(input_shape=INPUT_SHAPE, batch_size=None):
    if batch_size is not None:
        inputs = layers.Input(batch_shape=(batch_size,) + tuple(input_shape), name="frames")
    else:
        inputs = layers.Input(shape=input_shape, name="frames")

    # Strided conv backbone (no MaxPool — better for INT8 / ESP-NN)
    x = layers.Conv2D(8, (5, 5), strides=2, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.Conv2D(16, (3, 3), strides=2, padding="same", activation="relu", name="conv2")(x)
    x = layers.Conv2D(24, (3, 3), strides=2, padding="same", activation="relu", name="conv3")(x)
    x = layers.Conv2D(32, (3, 3), strides=2, padding="same", activation="relu", name="conv4")(x)
    x = layers.Conv2D(32, (3, 3), strides=2, padding="same", activation="relu", name="conv5")(x)

    # GAP instead of Flatten keeps parameter count near the paper (~23k)
    # At 24x24 with 5 stride-2 layers, spatial collapses to 1x1 — same as Flatten
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.15)(x)
    shared = layers.Dense(32, activation="relu", name="shared_dense")(x)
    shared = layers.Dense(16, activation="relu", name="shared_dense2")(shared)

    # Linear logits; activations applied in loss / post-processing for stable INT8 export
    steering = layers.Dense(1, name="steering")(shared)
    throttle = layers.Dense(1, name="throttle")(shared)

    # Single concatenated head for EloquentTinyML / single-output TFLite Micro
    outputs = layers.Concatenate(name="output")([steering, throttle])

    return models.Model(inputs=inputs, outputs=outputs, name="TinyNav_CNN")


if __name__ == "__main__":
    model = build_model()
    model.summary()

    total_params = model.count_params()
    print("\n==========================================")
    print(f"Total Parameters: {total_params:,}")
    print(f"Input shape: {INPUT_SHAPE}")
    if total_params <= 45000:
        print("STATUS: OK — within TinyML budget (<45k)")
    else:
        print("STATUS: WARN — over 45k params, consider thinner layers")
    print("==========================================\n")

    model.compile(optimizer="adam", loss="mse")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(BASE_MODEL_FILE)
    print(f"Baseline saved to: {BASE_MODEL_FILE}")
