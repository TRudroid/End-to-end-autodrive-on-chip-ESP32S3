"""
Post-training INT8 quantization + C header export for TFLite Micro / EloquentTinyML.
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from train_model import CUSTOM_OBJECTS
from tinynav_config import (
    BEST_MODEL_FILE,
    PREPROCESSED_FILE,
    TFLITE_FILE,
    MODEL_HEADER_FILE,
    FIRMWARE_HEADER_STREAM,
    FIRMWARE_HEADER_ONBOARD,
    INPUT_SHAPE,
    MODELS_DIR,
)


def build_export_model(input_shape=INPUT_SHAPE):
    """Inference graph for old EloquentTinyML (no MEAN / no Dropout / no Concatenate).

    GlobalAveragePooling2D becomes MEAN (unsupported). Use AveragePooling2D instead
    → AVERAGE_POOL_2D, which AllOpsResolver in bundled EloquentTinyML does register.
    """
    inputs = layers.Input(shape=input_shape, batch_size=1, name="frames")
    x = layers.Conv2D(8, (5, 5), strides=2, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.Conv2D(16, (3, 3), strides=2, padding="same", activation="relu", name="conv2")(x)
    x = layers.Conv2D(24, (3, 3), strides=2, padding="same", activation="relu", name="conv3")(x)
    x = layers.Conv2D(32, (3, 3), strides=2, padding="same", activation="relu", name="conv4")(x)
    x = layers.Conv2D(32, (3, 3), strides=2, padding="same", activation="relu", name="conv5")(x)
    # 24x24 → … → 1x1x32 after 5 stride-2 convs; Flatten ≡ GAP at 1x1 (no MEAN op)
    x = layers.Flatten(name="gap")(x)
    shared = layers.Dense(32, activation="relu", name="shared_dense")(x)
    shared = layers.Dense(16, activation="relu", name="shared_dense2")(shared)
    outputs = layers.Dense(2, name="output")(shared)
    return models.Model(inputs=inputs, outputs=outputs, name="TinyNav_Export")


def transfer_weights(trained, export_model):
    """Copy conv/shared weights; merge dual heads into Dense(2)."""
    name_map = [
        "conv1",
        "conv2",
        "conv3",
        "conv4",
        "conv5",
        "shared_dense",
        "shared_dense2",
    ]
    for name in name_map:
        export_model.get_layer(name).set_weights(trained.get_layer(name).get_weights())

    w_s, b_s = trained.get_layer("steering").get_weights()
    w_t, b_t = trained.get_layer("throttle").get_weights()
    # Dense(2): columns = [steer, throttle]
    kernel = np.concatenate([w_s, w_t], axis=1)
    bias = np.concatenate([b_s, b_t], axis=0)
    export_model.get_layer("output").set_weights([kernel, bias])


def representative_data_gen():
    data = np.load(PREPROCESSED_FILE)
    x_train = data["x_train"]
    n = min(len(x_train), 500)
    for i in range(n):
        yield [x_train[i : i + 1].astype(np.float32) / 255.0]


def export_c_header(tflite_bytes, path):
    hex_bytes = [f"0x{b:02x}" for b in tflite_bytes]
    content = f"""// Auto-generated TinyNav INT8 model (do not edit)
#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <cstdint>

alignas(16) const unsigned char model_data[] = {{
    {', '.join(hex_bytes)}
}};
const unsigned int model_data_len = {len(tflite_bytes)};

const unsigned char* g_model = model_data;
const int g_model_len = (int)model_data_len;

#endif // MODEL_DATA_H
"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    if not os.path.exists(BEST_MODEL_FILE):
        print(f"Missing {BEST_MODEL_FILE}. Train first.")
        return
    if not os.path.exists(PREPROCESSED_FILE):
        print(f"Missing {PREPROCESSED_FILE}. Preprocess first.")
        return

    print(f"Loading {BEST_MODEL_FILE}...")
    trained = tf.keras.models.load_model(BEST_MODEL_FILE, custom_objects=CUSTOM_OBJECTS)

    print("Building TFLite-friendly export graph (no Dropout/Concatenate)...")
    export_model = build_export_model()
    transfer_weights(trained, export_model)
    export_model.trainable = False

    print("Converting to full INT8 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(TFLITE_FILE, "wb") as f:
        f.write(tflite_model)
    print(f"Saved {TFLITE_FILE} ({len(tflite_model) / 1024:.1f} KB)")

    export_c_header(tflite_model, MODEL_HEADER_FILE)
    print(f"Saved {MODEL_HEADER_FILE}")

    for dest in (FIRMWARE_HEADER_ONBOARD, FIRMWARE_HEADER_STREAM):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(MODEL_HEADER_FILE, dest)
        print(f"Copied -> {dest}")

    shutil.copy2(MODEL_HEADER_FILE, "model_data.h")
    print("Quantization complete — flash esp32_firmware_onboard with model_data.h")


if __name__ == "__main__":
    main()
