import os

# Project Paths

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset", "chest_xray")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Image Settings

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

BATCH_SIZE = 32
SEED = 42

# Training Settings

EPOCHS = 10
LEARNING_RATE = 0.001

# Model Save Path

MODEL_SAVE_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "pneumonia_cnn.keras"
)