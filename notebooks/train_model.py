"""
Train TinyNav CNN (steering + throttle) with on-the-fly augmentation.
"""

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tinynav_config import PREPROCESSED_FILE, BASE_MODEL_FILE, BEST_MODEL_FILE, INPUT_SHAPE

BATCH_SIZE = 16
# Cap cao: EarlyStopping sẽ cắt khi val_loss không cải thiện
EPOCHS = 500
EARLY_STOP_PATIENCE = 40          # cho phép plateau dài trước khi dừng
EARLY_STOP_MIN_DELTA = 1e-5       # cải thiện tối thiểu để tính là tốt hơn
REDUCE_LR_PATIENCE = 12
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6
INITIAL_LR = 1e-3


def data_generator(X, Y_steer, Y_throt, batch_size=16, augment=True):
    n = len(X)
    while True:
        indices = np.arange(n)
        if augment:
            np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            x = X[batch_idx].astype(np.float32) / 255.0
            ys = Y_steer[batch_idx].copy()
            yt = Y_throt[batch_idx].copy()

            if augment:
                for i in range(len(x)):
                    # Horizontal flip (paper-style directional augmentation)
                    if np.random.rand() > 0.5:
                        x[i] = np.flip(x[i], axis=1)
                        ys[i] = -ys[i]

                    # Small lateral shift with steering compensation
                    dx = np.random.randint(-10, 11)
                    if dx > 0:
                        x[i][:, dx:, :] = x[i][:, :-dx, :]
                        x[i][:, :dx, :] = x[i][:, dx : dx + 1, :]
                    elif dx < 0:
                        x[i][:, :dx, :] = x[i][:, -dx:, :]
                        x[i][:, dx:, :] = x[i][:, dx - 1 : dx, :]
                    ys[i] = np.clip(ys[i] + dx * 0.04, -1.0, 1.0)

                    brightness = np.random.uniform(0.8, 1.2)
                    x[i] = np.clip(x[i] * brightness, 0.0, 1.0)

                    contrast = np.random.uniform(0.85, 1.15)
                    mean = np.mean(x[i])
                    x[i] = np.clip((x[i] - mean) * contrast + mean, 0.0, 1.0)

                    if np.random.rand() > 0.5:
                        noise = np.random.normal(0.0, np.random.uniform(0.01, 0.03), x[i].shape)
                        x[i] = np.clip(x[i] + noise, 0.0, 1.0)

            y = np.column_stack([ys, yt]).astype(np.float32)
            yield x, y


def custom_loss(y_true, y_pred):
    y_s = y_true[:, 0:1]
    y_t = y_true[:, 1:2]
    p_s = tf.tanh(y_pred[:, 0:1])
    p_t = tf.sigmoid(y_pred[:, 1:2])
    return tf.reduce_mean(tf.square(y_s - p_s)) + tf.reduce_mean(tf.square(y_t - p_t))


def steer_mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[:, 0:1] - tf.tanh(y_pred[:, 0:1])))


def throt_mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[:, 1:2] - tf.sigmoid(y_pred[:, 1:2])))


CUSTOM_OBJECTS = {
    "custom_loss": custom_loss,
    "steer_mae": steer_mae,
    "throt_mae": throt_mae,
}


def main():
    if not os.path.exists(PREPROCESSED_FILE):
        print(f"Missing {PREPROCESSED_FILE}. Run notebooks/preprocess_dataset.py first.")
        return

    data = np.load(PREPROCESSED_FILE)
    x_train = data["x_train"]
    ys_train, yt_train = data["y_train_steer"], data["y_train_throt"]
    x_val = data["x_val"]
    ys_val, yt_val = data["y_val_steer"], data["y_val_throt"]
    x_test = data["x_test"]
    ys_test, yt_test = data["y_test_steer"], data["y_test_throt"]
    print(f"Loaded train={len(x_train)} val={len(x_val)} test={len(x_test)}")

    # Chỉ resume nếu architecture khớp TinyNav (24x24x20 → 2 outputs)
    model = None
    if os.path.exists(BEST_MODEL_FILE):
        try:
            candidate = tf.keras.models.load_model(BEST_MODEL_FILE, custom_objects=CUSTOM_OBJECTS)
            in_ok = tuple(candidate.input_shape[1:]) == tuple(INPUT_SHAPE)
            out_ok = int(candidate.output_shape[-1]) == 2
            if in_ok and out_ok:
                print(f"Resuming from compatible {BEST_MODEL_FILE}")
                model = candidate
            else:
                print(
                    f"Ignoring incompatible {BEST_MODEL_FILE} "
                    f"(in={candidate.input_shape}, out={candidate.output_shape}); starting from baseline."
                )
                bak = BEST_MODEL_FILE.replace(".keras", "_legacy.keras")
                os.replace(BEST_MODEL_FILE, bak)
                print(f"Moved old checkpoint -> {bak}")
        except Exception as e:
            print(f"Could not load {BEST_MODEL_FILE} ({e}); starting from baseline.")

    if model is None:
        if not os.path.exists(BASE_MODEL_FILE):
            print(f"Missing {BASE_MODEL_FILE}. Run notebooks/design_model.py first.")
            return
        print(f"Loading baseline {BASE_MODEL_FILE}")
        model = tf.keras.models.load_model(BASE_MODEL_FILE)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(INITIAL_LR),
        loss=custom_loss,
        metrics=[steer_mae, throt_mae],
    )

    train_gen = data_generator(x_train, ys_train, yt_train, BATCH_SIZE, True)
    val_gen = data_generator(x_val, ys_val, yt_val, BATCH_SIZE, False)
    train_steps = int(np.ceil(len(x_train) / BATCH_SIZE))
    val_steps = int(np.ceil(len(x_val) / BATCH_SIZE))

    print(
        f"Training up to {EPOCHS} epochs | early_stop patience={EARLY_STOP_PATIENCE} "
        f"| reduce_lr patience={REDUCE_LR_PATIENCE}"
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            min_delta=EARLY_STOP_MIN_DELTA,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_delta=EARLY_STOP_MIN_DELTA,
            min_lr=MIN_LR,
            verbose=1,
        ),
        ModelCheckpoint(
            BEST_MODEL_FILE,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )

    # Persist best weights explicitly (EarlyStopping already restored them)
    model.save(BEST_MODEL_FILE)
    stopped = len(history.history.get("loss", []))
    print(f"Finished after {stopped}/{EPOCHS} epochs (best weights restored & saved).")

    y_test = np.column_stack([ys_test, yt_test])
    results = model.evaluate(x_test.astype(np.float32) / 255.0, y_test, verbose=0)
    print("\n=== TEST METRICS ===")
    for name, val in zip(model.metrics_names, results):
        print(f"{name}: {val:.4f}")
    print("====================\n")


if __name__ == "__main__":
    main()
