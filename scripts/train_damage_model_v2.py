import os
import shutil
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# =====================================================
# PATHS
# =====================================================
CSV_PATH = "data/raw/train.csv"
IMAGE_FOLDER = "data/raw/images"
DATASET_FOLDER = "data/damage_images"
MODEL_FOLDER = "models"
MODEL_PATH = "models/damage_classifier_v2.keras"

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)

# =====================================================
# LABELS
# =====================================================

label_map = {
    1: "Crack",
    2: "Scratch",
    3: "Flat_tire",
    4: "Dent",
    5: "Glass_shatter",
    6: "Lamp_broken"
}

# =====================================================
# PREPARE DATASET
# =====================================================

def prepare_dataset():

    print("\nPreparing dataset...\n")

    df = pd.read_csv(
        CSV_PATH
    )

    for class_name in label_map.values():
        os.makedirs(
            os.path.join(
                DATASET_FOLDER,
                class_name
            ),
            exist_ok=True
        )

    copied = 0

    for _, row in df.iterrows():
        src = os.path.join(
            IMAGE_FOLDER,
            row["filename"]
        )

        class_name = label_map[row["label"]]

        dst = os.path.join(
            DATASET_FOLDER,
            class_name,
            row["filename"]
        )

        if os.path.exists(src):
            shutil.copy2(
                src,
                dst
            )

            copied += 1

    print(
        f"Copied images: {copied}"
    )


# =====================================================
# DATASETS
# =====================================================

def create_datasets():

    train_ds = tf.keras.utils.image_dataset_from_directory(

        DATASET_FOLDER,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(224,224),
        batch_size=32
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(

        DATASET_FOLDER,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(224,224),
        batch_size=32
    )

    class_names = train_ds.class_names

    print("\nCLASS ORDER:")
    print(class_names)

    with open(
        "models/class_names.json",
        "w"
    ) as f:

        json.dump(
            class_names,
            f
        )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(
        AUTOTUNE
    )

    val_ds = val_ds.prefetch(
        AUTOTUNE
    )

    return train_ds, val_ds, class_names

# =====================================================
# CLASS WEIGHTS
# =====================================================

def calculate_class_weights(train_ds):
    labels = []

    for _, y in train_ds:
        labels.extend(
            y.numpy()
        )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )

    class_weights = dict(
        enumerate(weights)
    )

    print(
        "\nClass weights:"
    )

    print(
        class_weights
    )

    return class_weights

# =====================================================
# MODEL
# =====================================================

def build_model(num_classes):

    augmentation = tf.keras.Sequential([

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.1
        ),

        layers.RandomZoom(
            0.1
        )
    ])

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    model = models.Sequential([

        augmentation,

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dropout(
            0.4
        ),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(
            0.3
        ),

        layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    return model, base_model

# =====================================================
# TRAIN
# =====================================================

def train():

    prepare_dataset()
    train_ds, val_ds, class_names = create_datasets()

    class_weights = calculate_class_weights(
        train_ds
    )

    model, base_model = build_model(
        len(class_names)
    )

    callbacks = [

        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max"
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2
        )
    ]

    print("\nStage 1: Training classifier head\n")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks,
        class_weight=class_weights
    )

    # =================================================
    # FINE TUNING
    # =================================================
    print("\nStage 2: Fine tuning EfficientNet\n")

    base_model.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.00001
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks,
        class_weight=class_weights
    )

    print("\nTraining finished")
    print(
        "Saved model:"
    )
    print(
        MODEL_PATH
    )

if __name__ == "__main__":

    train()