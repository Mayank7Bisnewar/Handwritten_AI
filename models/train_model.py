"""
Handwritten Notes AI — models/train_model.py
CNN + LSTM handwritten text recognition model trainer
Uses IAM Handwriting Dataset (https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)

Usage:
    python models/train_model.py --data dataset/iam_words --epochs 50

This script:
  1. Loads word images + labels from IAM dataset
  2. Preprocesses images (grayscale, resize, normalise)
  3. Builds a CNN-LSTM model with CTC loss
  4. Trains and saves the model to models/handwritten_model.h5
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
IMG_HEIGHT   = 32      # fixed input height
IMG_WIDTH    = 128     # fixed input width
MAX_LABEL_LEN = 32     # max word length
CHARSET = " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

char2idx = {c: i + 1 for i, c in enumerate(CHARSET)}   # 0 = blank for CTC
idx2char = {i + 1: c for i, c in enumerate(CHARSET)}

NUM_CLASSES = len(CHARSET) + 1  # +1 for CTC blank


# ─────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────

def load_iam_words(data_dir: str):
    """
    Load IAM word-level images and labels from the words/ subdirectory.
    Expected layout:
        data_dir/
            words.txt  (annotations)
            words/
                a01/a01-000u/a01-000u-00-00.png  …
    """
    ann_path = os.path.join(data_dir, "words.txt")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"words.txt not found in {data_dir}")

    images, labels = [], []
    with open(ann_path) as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split(" ")
            if parts[1] != "ok":
                continue
            word_id = parts[0]
            text    = parts[-1]

            # Build path: a01-000u-00-00 → words/a01/a01-000u/a01-000u-00-00.png
            segs     = word_id.split("-")
            img_path = os.path.join(data_dir, "words",
                                    segs[0], f"{segs[0]}-{segs[1]}",
                                    word_id + ".png")
            if not os.path.exists(img_path):
                continue

            images.append(img_path)
            labels.append(text)

    return images, labels


def preprocess_image(path: str) -> np.ndarray:
    """Load, grayscale, resize and normalise a word image."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, -1)   # (H, W, 1)
    return img


def encode_label(text: str) -> np.ndarray:
    encoded = [char2idx.get(c, 0) for c in text[:MAX_LABEL_LEN]]
    # Pad to MAX_LABEL_LEN
    padded  = encoded + [0] * (MAX_LABEL_LEN - len(encoded))
    return np.array(padded, dtype=np.int32)


def build_dataset(data_dir: str, val_split: float = 0.1):
    image_paths, labels = load_iam_words(data_dir)
    images_arr, labels_arr = [], []

    for path, label in zip(image_paths, labels):
        img = preprocess_image(path)
        if img is None:
            continue
        images_arr.append(img)
        labels_arr.append(encode_label(label))

    X = np.array(images_arr)                    # (N, H, W, 1)
    y = np.array(labels_arr)                    # (N, MAX_LABEL_LEN)

    split = int(len(X) * (1 - val_split))
    return (X[:split], y[:split]), (X[split:], y[split:])


# ─────────────────────────────────────────────────────────────────
# Model — CNN encoder + bidirectional LSTM + CTC head
# ─────────────────────────────────────────────────────────────────

def build_model() -> keras.Model:
    input_img = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")
    labels    = keras.Input(shape=(MAX_LABEL_LEN,), name="label", dtype=tf.int32)

    # ── CNN feature extractor ──────────────────────────────────
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 1))(x)           # keep width
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Conv2D(256, (2, 2), activation="relu")(x)

    # Squeeze height → (batch, time_steps, features)
    new_shape = (x.shape[-2], x.shape[-1] * x.shape[-3]) if x.shape[-3] else (-1, x.shape[-1])
    x = layers.Reshape(target_shape=(x.shape[2], x.shape[1] * x.shape[3]))(x)

    # ── Bidirectional LSTM ─────────────────────────────────────
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    # ── CTC output ─────────────────────────────────────────────
    output = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    # CTC loss layer
    input_length  = keras.Input(shape=(1,), name="input_length",  dtype=tf.int32)
    label_length  = keras.Input(shape=(1,), name="label_length",  dtype=tf.int32)

    ctc_loss = layers.Lambda(
        lambda args: tf.keras.backend.ctc_batch_cost(*args),
        name="ctc_loss"
    )([labels, output, input_length, label_length])

    model = keras.Model(
        inputs=[input_img, labels, input_length, label_length],
        outputs=ctc_loss,
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=lambda y_true, y_pred: y_pred)

    # Prediction model (no CTC inputs)
    pred_model = keras.Model(inputs=input_img, outputs=output, name="prediction_model")
    return model, pred_model


# ─────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────

def train(data_dir: str, epochs: int = 50, batch_size: int = 32):
    print("📦 Loading IAM dataset …")
    (X_train, y_train), (X_val, y_val) = build_dataset(data_dir)
    print(f"   Train: {len(X_train)} samples | Val: {len(X_val)} samples")

    # Sequence lengths (number of time steps after CNN)
    # For our architecture: width // 4 after two MaxPool with pool_size=(1,1)
    time_steps = IMG_WIDTH // 4
    train_input_len = np.full((len(X_train), 1), time_steps, dtype=np.int32)
    val_input_len   = np.full((len(X_val),   1), time_steps, dtype=np.int32)
    train_label_len = np.sum(y_train > 0, axis=1, keepdims=True).astype(np.int32)
    val_label_len   = np.sum(y_val   > 0, axis=1, keepdims=True).astype(np.int32)

    train_inputs = {
        "image":        X_train,
        "label":        y_train,
        "input_length": train_input_len,
        "label_length": train_label_len,
    }
    val_inputs = {
        "image":        X_val,
        "label":        y_val,
        "input_length": val_input_len,
        "label_length": val_label_len,
    }
    dummy_out = np.zeros(len(X_train))
    dummy_val = np.zeros(len(X_val))

    model, pred_model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "models/handwritten_model_best.h5",
            save_best_only=True, monitor="val_loss", verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, restore_best_weights=True
        ),
    ]

    print(f"\n🚀 Training for {epochs} epochs …")
    model.fit(
        train_inputs, dummy_out,
        validation_data=(val_inputs, dummy_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    pred_model.save("models/handwritten_model.h5")
    print("✅ Prediction model saved to models/handwritten_model.h5")


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train handwritten text recognition model")
    parser.add_argument("--data",       default="dataset/iam_words", help="Path to IAM word dataset")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch_size)
