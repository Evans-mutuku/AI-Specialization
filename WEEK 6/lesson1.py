# Example: Deploying a lightweight AI model on an edge device (e.g., Raspberry Pi)
import tensorflow as tf
import numpy as np

# Load a pre-trained lightweight model (e.g., MobileNet)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Simulate input data (e.g., image from a camera)
image = np.random.rand(224, 224, 3)
image = tf.keras.applications.mobilenet_v2.preprocess_input(image[np.newaxis, ...])

predictions = model.predict(image)
decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)
print("Predictions:", decoded_predictions)