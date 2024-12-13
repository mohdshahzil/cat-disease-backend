import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model (ensure it's in the same directory or provide the correct path)
model = load_model('cat_skin_disease_model.h5')

# Define the class labels
class_labels = {0: 'Flea_Allergy', 1: 'Healthy', 2: 'Ringworm', 3: 'Scabies'}

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the image size for model input
IMG_SIZE = (150, 150)

# Route to check if the server is running
@app.route('/')
def home():
    return "Cat Skin Disease Prediction API is running!"

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file to the upload folder
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess the image for prediction
    img = image.load_img(filepath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make the prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the index of the highest prediction

    # Get the predicted class label
    predicted_label = class_labels[predicted_class]

    # Return the prediction as JSON
    return jsonify({
        'predicted_class': predicted_label,
        'prediction_score': float(predictions[0][predicted_class])
    })

if __name__ == '__main__':
    app.run(debug=True)
