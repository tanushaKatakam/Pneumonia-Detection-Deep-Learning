import tensorflow as tf
import matplotlib.pyplot as plt

from dataset import load_datasets
from model import build_model
from config import EPOCHS, MODEL_SAVE_PATH

import pandas as pd
import os
from config import BASE_DIR

# Load Dataset

train_dataset, val_dataset, test_dataset, class_names = load_datasets()

class_weight = {
    0: 1.44,
    1: 0.50
}

print("\nClasses:", class_names)

# Build Model

model = build_model()

model.summary()

# Callbacks

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# Train Model

os.makedirs(os.path.join(BASE_DIR, "saved_models"), exist_ok=True)

os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop, reduce_lr],
    class_weight=class_weight,
    verbose=2
)

# Save Training History


history_df = pd.DataFrame(history.history)

history_df.to_csv(
    os.path.join(BASE_DIR, "outputs", "training_history.csv"),
    index=False
)

# Evaluate Model

test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_dataset)

print(f"\nTest Loss      : {test_loss:.4f}")
print(f"Test Accuracy  : {test_accuracy:.4f}")
print(f"Test Precision : {test_precision:.4f}")
print(f"Test Recall    : {test_recall:.4f}")

# Accuracy Graph

plt.figure(figsize=(10,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.savefig(
    os.path.join(BASE_DIR, "outputs", "accuracy.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Loss Graph

plt.figure(figsize=(10,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.savefig(
    os.path.join(BASE_DIR, "outputs", "loss.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()