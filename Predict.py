import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import joblib
import matplotlib.pyplot as plt

# load model
class WeedClassifier:
    ALLOWED_EXTENSIONS = ('.jpg', '.jpeg')  # Only allow these

    def __init__(self, model_path='weed_species_classifier_vgg16.h5',
                 encoder_path='species_label_encoder.pkl'):
        self.model = load_model(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.img_size = (224, 224)  # Must match training size

    # preprocess
    def preprocess_image(self, img_path):
        img = load_img(img_path, target_size=self.img_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    # predict single image
    def predict(self, img_path, show_image=True):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")
        if not img_path.lower().endswith(self.ALLOWED_EXTENSIONS):
            raise ValueError(f"Only JPG or JPEG images are allowed: {img_path}")

        # Preprocess and predict
        processed_img = self.preprocess_image(img_path)
        predictions = self.model.predict(processed_img)
        class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        class_name = self.label_encoder.inverse_transform([class_idx])[0]

        # Display results
        if show_image:
            self._display_prediction(img_path, class_name, confidence)

        return class_name, float(confidence)

    # batch predict in a folder
    def batch_predict(self, img_dir):
        results = []
        for filename in os.listdir(img_dir):
            if filename.lower().endswith(self.ALLOWED_EXTENSIONS):
                img_path = os.path.join(img_dir, filename)
                try:
                    class_name, confidence = self.predict(img_path, show_image=False)
                    results.append((filename, class_name, confidence))
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        return results

    # show image with prediction
    def _display_prediction(self, img_path, class_name, confidence):
        img = load_img(img_path, target_size=(300, 300))
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"Prediction: {class_name}\nConfidence: {confidence:.2%}")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    classifier = WeedClassifier()

    # Single image prediction
    test_image = "Negative14.jpeg"
    if os.path.exists(test_image):
        try:
            class_name, confidence = classifier.predict(test_image)
            print(f"\nPrediction for {test_image}:")
            print(f"Class: {class_name} ({confidence:.2%})")
        except ValueError as ve:
            print(ve)
    else:
        print(f"\nTest image not found at {test_image}")

    # Batch prediction
    test_dir = "test_images"
    if os.path.isdir(test_dir):
        print(f"\nBatch predicting on images in {test_dir}:")
        results = classifier.batch_predict(test_dir)
        for filename, class_name, confidence in results:
            print(f"{filename}: {class_name} ({confidence:.2%})")
