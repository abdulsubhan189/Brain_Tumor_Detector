import os
import logging
import tempfile
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from urllib.request import urlopen

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

BINARY_MODEL_PATH = "models/binary_model.tflite"
MULTI_MODEL_PATH = "models/multi_model.tflite"

CLASSES = ["glioma", "meningioma", "pituitary"]

BINARY_SIZE = 224
MULTI_SIZE = 384

app = Flask(__name__)
CORS(app) 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

logger.info("Loading TFLite models...")
binary_interpreter = load_tflite_model(BINARY_MODEL_PATH)
multi_interpreter = load_tflite_model(MULTI_MODEL_PATH)
logger.info("Models loaded successfully")


def preprocess_image(image_path):
    """
    Reads an image, resizes it to both binary and multi input sizes,
    normalises to [0,1], and returns two batches (1, H, W, 3).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image file")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Binary preprocessing (224x224)
    bin_img = cv2.resize(img, (BINARY_SIZE, BINARY_SIZE))
    bin_img = bin_img.astype(np.float32) / 255.0
    bin_img = np.expand_dims(bin_img, axis=0)

    # Multi preprocessing (384x384)
    multi_img = cv2.resize(img, (MULTI_SIZE, MULTI_SIZE))
    multi_img = multi_img.astype(np.float32) / 255.0
    multi_img = np.expand_dims(multi_img, axis=0)

    return bin_img, multi_img


def predict_binary(img_batch):
    """Returns probability of tumor (class 1)."""
    input_details = binary_interpreter.get_input_details()
    output_details = binary_interpreter.get_output_details()

    binary_interpreter.set_tensor(input_details[0]['index'], img_batch)
    binary_interpreter.invoke()
    output = binary_interpreter.get_tensor(output_details[0]['index'])
    return float(output[0][0])  # sigmoid output

def predict_multi(img_batch):
    """Returns softmax probabilities for 3 classes."""
    input_details = multi_interpreter.get_input_details()
    output_details = multi_interpreter.get_output_details()

    multi_interpreter.set_tensor(input_details[0]['index'], img_batch)
    multi_interpreter.invoke()
    output = multi_interpreter.get_tensor(output_details[0]['index'])[0]
    return output  # shape (3,)


def is_mri_like(img_batch):
    """
    Very basic check to reject obviously non‑MRI images.
    Returns True if the image looks like an MRI scan.
    """
    img = (img_batch[0] * 255).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size
    if edge_ratio > 0.35:
        return False

    std = np.std(gray)
    if std < 8:
        return False

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / (hist.sum() + 1e-6)
    entropy = -np.sum(hist * np.log2(hist + 1e-6))
    if entropy > 7.5:
        return False

    return True


def get_prediction(bin_img, multi_img):
    """
    Runs binary check, then if tumor detected runs multi‑class.
    Returns a dict ready for JSON response.
    """
    if not is_mri_like(bin_img):
        return {
            "label": "Not a valid MRI scan",
            "confidence": 0,
            "all_predictions": None,
            "error": "Image does not look like an MRI"
        }

    tumor_prob = predict_binary(bin_img)

    if tumor_prob < 0.5:
        return {
            "label": "No Tumor",
            "confidence": round((1 - tumor_prob) * 100, 2),
            "all_predictions": None
        }

    probs = predict_multi(multi_img)
    max_idx = int(np.argmax(probs))
    max_conf = float(probs[max_idx])

    if max_conf < 0.70:
        return {
            "label": "Uncertain / Invalid MRI",
            "confidence": round(max_conf * 100, 2),
            "all_predictions": None,
            "error": "Low confidence in classification"
        }

    all_preds = [
        {
            "label": CLASSES[i].capitalize(),
            "confidence": round(float(probs[i]) * 100, 2)
        }
        for i in range(len(CLASSES))
    ]

    return {
        "label": CLASSES[max_idx].capitalize(),
        "confidence": round(max_conf * 100, 2),
        "all_predictions": all_preds
    }

@app.route("/")
def index():
    """Serve the frontend HTML page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    file = request.files.get("file")
    url = request.form.get("url")
    temp_path = None

    try:
        if file:
            filename = secure_filename(file.filename)
            temp_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(temp_path)

        elif url:
            # Download image from URL
            with urlopen(url) as response:
                img_data = response.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(img_data)
                temp_path = tmp.name

        else:
            return jsonify({
                "label": "Error",
                "confidence": 0,
                "error": "No input provided"
            })

        bin_img, multi_img = preprocess_image(temp_path)
        result = get_prediction(bin_img, multi_img)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "label": "Error",
            "confidence": 0,
            "error": str(e)
        })

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)