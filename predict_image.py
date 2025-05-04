import constants as con
import Predict as pre
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import visualize_prediction as Visual

def predict_image(img_path):
    img = Visual.image.load_img(img_path, target_size=con.IMG_SIZE)
    img_array = Visual.image.img_to_array(img)

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Rescale image
    img_array = img_array / 255.0
    predictions = pre.model.predict(img_array)

    # Get the predicted
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    return predicted_class_idx, confidence_score
