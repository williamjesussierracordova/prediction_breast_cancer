import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from PIL import Image
from flask import Flask, request, jsonify
import requests
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load and preprocess data
def load_data():
    X = np.load('X.npy')
    Y = np.load('Y.npy')
    X = X.astype('float32') / 255.0
    Y = Y.astype('int')
    return X, Y

# Define and train model
def train_model(X, Y):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
    class_weight_dict = dict(enumerate(class_weights))

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    model.fit(
        X_train, Y_train,
        batch_size=32,
        epochs=30,
        validation_data=(X_val, Y_val),
        class_weight=class_weight_dict,
        callbacks=[tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6)]
    )

    return model

# Preprocess a single image
def preprocess_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((50, 50))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict on a single image
def predict_single_image(image_url, model):
    processed_image = preprocess_image(image_url)
    prediction = model.predict(processed_image)[0][0]
    class_prediction = "Cancer" if prediction > 0.5 else "No Cancer"
    confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)
    return class_prediction, confidence

# Train and save model if it doesn't exist
model_path = 'breast_cancer_model.h5'
if not os.path.exists(model_path):
    X, Y = load_data()
    model = train_model(X, Y)
    model.save(model_path)
else:
    model = tf.keras.models.load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'image_url' not in data:
        return jsonify({'error': 'No image URL provided'}), 400
    
    image_url = data['image_url']
    try:
        class_prediction, confidence = predict_single_image(image_url, model)
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 400
    
    return jsonify({
        'prediction': class_prediction,
        'confidence': confidence
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)