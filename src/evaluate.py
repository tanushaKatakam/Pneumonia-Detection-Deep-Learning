import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from dataset import load_datasets
from config import MODEL_SAVE_PATH, BASE_DIR

# Load dataset
_, _, test_dataset, class_names = load_datasets()

# Load trained model
model = tf.keras.models.load_model(MODEL_SAVE_PATH)

# Evaluate
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_dataset)

print("\n========== Test Metrics ==========")

print(f"Loss      : {test_loss:.4f}")
print(f"Accuracy  : {test_accuracy:.4f}")
print(f"Precision : {test_precision:.4f}")
print(f"Recall    : {test_recall:.4f}")

# Predictions
predictions = model.predict(test_dataset)

predicted_labels = (predictions > 0.5).astype(int).flatten()

true_labels = np.concatenate(
    [labels.numpy().flatten() for _, labels in test_dataset]
)

print("\n========== Classification Report ==========\n")

print(classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names
))

# Confusion Matrix
cm = confusion_matrix(
    true_labels,
    predicted_labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Blues")

os.makedirs(
    os.path.join(BASE_DIR, "outputs"),
    exist_ok=True
)

plt.savefig(
    os.path.join(
        BASE_DIR,
        "outputs",
        "confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()