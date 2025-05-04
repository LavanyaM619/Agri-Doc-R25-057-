import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt

# Constants
MODEL_PATH = "final_pest_model.h5"
IMG_PATH = "3.jpg"
IMG_SIZE = (224, 224)
TEST_PATH = "PestDataset/test"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# Get class names from folder structure
class_names = sorted(os.listdir(TEST_PATH))
class_names = [folder for folder in class_names if os.path.isdir(os.path.join(TEST_PATH, folder))]
print(f"Classes detected: {class_names}")

# Load and preprocess the image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img, img_array

# Predict
img_loaded, img_array = load_and_preprocess_image(IMG_PATH)
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions[0])
confidence = np.max(predictions[0])

# Map index to class label
if predicted_class_index < len(class_names):
    predicted_class_label = class_names[predicted_class_index]
    print(f"Predicted Class: {predicted_class_label} )")
else:
    predicted_class_label = "Unknown"
    print(f"Predicted index {predicted_class_index} is out of range for class labels.")

# Display the image with prediction
plt.figure(figsize=(6, 6))
plt.imshow(img_loaded)
plt.axis('off')
plt.title(f"Prediction: {predicted_class_label}", fontsize=14)
plt.show()
