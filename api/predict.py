from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Adjust file paths to root directory (since predict.py is in api/)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'trained_model.keras')
LABELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'labels.txt')

# Load the trained model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load class labels
try:
    with open(LABELS_PATH, "r") as f:
        labels = f.read().splitlines()
except FileNotFoundError:
    labels = []
    print(f"Error: labels.txt not found")

# Preprocessing function
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((150, 150))
        img_array = np.array(img)
        # Normalize if model was trained with rescale=1./255
        # img_array = img_array / 255.0  # Uncomment if needed
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise Exception(f"Image preprocessing failed: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not labels:
        return jsonify({'error': 'Model or labels not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        image_bytes = request.files['image'].read()
        processed_image = preprocess_image(image_bytes)

        prediction = model.predict(processed_image)[0]
        top_index = int(np.argmax(prediction))
        confidence = float(prediction[top_index])
        label = labels[top_index]

        return jsonify({
            'label': label,
            'confidence': round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET', 'HEAD'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)