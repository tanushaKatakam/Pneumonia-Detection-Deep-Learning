import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from config import (
    MODEL_SAVE_PATH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH
)

# Load Trained Model


model = tf.keras.models.load_model(MODEL_SAVE_PATH)

# Class Labels

class_names = ["NORMAL", "PNEUMONIA"]

# Prediction Function

def predict_image(image_path):

    img = image.load_img(
        image_path,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
    )

    img_array = image.img_to_array(img)

    # Normalize pixels
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    probability = float(prediction[0][0])

    if probability >= 0.5:
        predicted_class = "PNEUMONIA"
        confidence = probability
    else:
        predicted_class = "NORMAL"
        confidence = 1 - probability

    print(f"\nPrediction : {predicted_class}")
    print(f"Confidence : {confidence*100:.2f}%")
    print(f"Pneumonia Probability : {probability*100:.2f}%")

    plt.imshow(img)
    plt.axis("off")
    plt.title(
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence:.2%}"
    )
    plt.show()

# Example

image_path = input("Enter image path: ")

predict_image(image_path)
