import tensorflow as tf

from config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED
)

AUTOTUNE = tf.data.AUTOTUNE


def load_datasets():

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="binary",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="binary",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="binary",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_dataset.class_names

    # Normalize pixels from [0,255] → [0,1]
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    train_dataset = train_dataset.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=AUTOTUNE
    )

    val_dataset = val_dataset.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=AUTOTUNE
    )

    test_dataset = test_dataset.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=AUTOTUNE
    )

    train_dataset = train_dataset.prefetch(AUTOTUNE)
    val_dataset = val_dataset.prefetch(AUTOTUNE)
    test_dataset = test_dataset.prefetch(AUTOTUNE)

    return train_dataset, val_dataset, test_dataset, class_names