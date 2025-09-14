from flask import Flask, request, jsonify
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import joblib
import tempfile
import traceback
import json

app = Flask(__name__)

# Parameters
IMG_WIDTH, IMG_HEIGHT = 224, 224  # Same as training
MODEL_PATH = "weed_species_classifier_vgg16.h5"
LABEL_ENCODER_PATH = "species_label_encoder.pkl"
UPLOAD_FOLDER = "temp_uploads"
CONTROL_DATA_PATH = "weed_controls.json"

# Allowed extensions
ALLOWED_EXTENSIONS = {"jpg", "jpeg"}


def allowed_file(filename):
    """Check if uploaded file has an allowed extension"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the pre-trained model and label encoder
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
if not os.path.exists(LABEL_ENCODER_PATH):
    raise FileNotFoundError(f"Label encoder file '{LABEL_ENCODER_PATH}' not found.")

model = load_model(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
print("Model and label encoder loaded successfully!")


def predict_species(img_path):
    """Predict weed species from image path"""
    if not os.path.exists(img_path):
        return None, 0.0

    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize

        # Prediction
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = float(np.max(predictions) * 100)

        return predicted_class, confidence

    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return None, 0.0


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": os.path.exists(MODEL_PATH),
        "label_encoder_loaded": os.path.exists(LABEL_ENCODER_PATH),
        "endpoint_type": "direct_image_upload"
    })


# Load control data from JSON
if not os.path.exists(CONTROL_DATA_PATH):
    raise FileNotFoundError(f"Control data file '{CONTROL_DATA_PATH}' not found.")

with open(CONTROL_DATA_PATH, "r") as f:
    control_data = json.load(f)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided in request"}), 400

        image_file = request.files["image"]

        if image_file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(image_file.filename):
            return jsonify({"error": "Only .jpg and .jpeg files are allowed"}), 400

        # Save uploaded file
        file_ext = os.path.splitext(image_file.filename)[1].lower() or ".jpg"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, dir=UPLOAD_FOLDER)
        image_file.save(temp_file.name)
        temp_file.close()

        # Prediction
        predicted_species, confidence = predict_species(temp_file.name)

        # Cleanup temp file
        try:
            os.remove(temp_file.name)
        except Exception as e:
            print(f"Warning: could not delete temp file {temp_file.name}: {e}")

        if predicted_species is None:
            return jsonify({"error": "Prediction failed"}), 500

        # Lookup control methods
        species_controls = control_data.get(predicted_species, {
            "description": "No control data available.",
            "control_methods": []
        })

        result = {
            "predicted_species": predicted_species,
            "confidence": round(confidence, 2),
            "description": species_controls["description"],
            "control_methods": species_controls["control_methods"],
            "status": "success"
        }

        return jsonify(result)

    except Exception as e:
        print(f"Server error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    print("Starting Weed Species Prediction API with direct image upload support...")
    app.run(host="0.0.0.0", port=5002, debug=True)
