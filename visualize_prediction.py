import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import constants as con

def visualize_prediction(img_path, predicted_class_name, confidence_score):
    # Load image for visualization
    img = image.load_img(img_path, target_size=con.IMG_SIZE)

    # Display the image
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.title(f"Predicted Class: {predicted_class_name}")
    plt.show()
