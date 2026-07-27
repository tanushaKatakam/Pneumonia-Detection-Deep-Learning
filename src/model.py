import tensorflow as tf
from config import IMAGE_HEIGHT, IMAGE_WIDTH, LEARNING_RATE


def build_model():

    # Data Augmentation

    data_augmentation = tf.keras.Sequential([

        tf.keras.layers.RandomRotation(0.05),

        tf.keras.layers.RandomZoom(0.10),

        tf.keras.layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        )

    ])

    # CNN Model

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        ),

        # Data Augmentation
        data_augmentation,

        # First Convolution Block
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3,3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.MaxPooling2D((2,2)),

        # Second Convolution Block
    
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3,3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.MaxPooling2D((2,2)),

        # Third Convolution Block
  
        tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.MaxPooling2D((2,2)),

        # Fully Connected Layers
      
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005)
        ),

        tf.keras.layers.Dropout(0.60),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )

    ])

    # Compile Model

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]

    )

    return model