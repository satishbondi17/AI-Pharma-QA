import os
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

DATA_DIR = Path("dataset")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 8
CLASS_NAMES = ["Broken", "Cracked", "Discolored", "Good"]
AUTOTUNE = tf.data.AUTOTUNE

def count_images(folder):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in extensions)

for split in ("train", "validation"):
    split_dir = DATA_DIR / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing folder: {split_dir.resolve()}")
    missing = [name for name in CLASS_NAMES if not (split_dir / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing class folders in {split}: {', '.join(missing)}")
    empty = [name for name in CLASS_NAMES if count_images(split_dir / name) == 0]
    if empty:
        raise ValueError(
            f"No supported images found in {split} folders: {', '.join(empty)}. "
            "Add JPG, JPEG, PNG, BMP, or GIF images."
        )

train_ds = tf.keras.utils.image_dataset_from_directory(
    str(DATA_DIR / "train"),
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=123
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    str(DATA_DIR / "validation"),
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump(CLASS_NAMES, f, indent=2)

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1.0 / 127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(len(CLASS_NAMES), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
model.save("tablet_model.keras")
print("Training complete. Saved tablet_model.keras")
print("WARNING: This sample dataset is synthetic and is only for testing the code pipeline.")
