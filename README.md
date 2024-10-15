# Breast Cancer Detection API

This project is an API developed in Python using TensorFlow, Keras, and Flask, which enables the detection of breast cancer from histological images. The model classifies images of size 50x50 pixels and returns the probability of cancer presence along with the class prediction.

## Requirements

- Python 3.8+
- TensorFlow
- Flask
- Flask-CORS
- scikit-learn
- numpy
- Pillow
- requests

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```
   
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that the data files X.npy and Y.npy are in the root directory of the project.

## Project Description

The project consists of the following steps:

1. Data Loading and Preprocessing:
   
  - Loads training data from .npy files.
  - Scales images between 0 and 1.
    
2. Model Definition and Training:
   
  - Uses a Convolutional Neural Network (CNN) for cancer detection.
  - Includes normalization, pooling, and dropout layers to improve accuracy and prevent overfitting.
  - Adjusts the model using class_weight to handle class imbalances.
    
3. Single Image Prediction:
   
  - Processes an image from a URL, resizes, and normalizes it for prediction.
  - The API returns the class prediction ("Cancer" or "No Cancer") and the associated confidence.
    
4. API Deployment:
   
  - The Flask API exposes a /predict endpoint that receives an image URL and returns the prediction.

## Usage

To make a breast cancer prediction, follow these steps:
1. Run the API locally:
  ```bash
  python app.py
  ```
2. Send a POST request to the /predict endpoint with the following format:
  ```json
  {
    "image_url": "IMAGE_URL"
  }
  ```
3. The response will have the following format:
  ```json
  {
    "prediction": "Cander",
    "confidence": 0.87
  }
  ```

## Model Training

The model is automatically trained if no breast_cancer_model.h5 file is found. If the file exists, the pretrained model will be loaded.

## Deployment in Production

To deploy the API in a production environment, you can use any hosting service that supports Flask, such as Render, Heroku, or AWS. Make sure to configure the appropriate environment variables for the port.

## Contributions

Contributions are welcome. Please fork the repository and submit a pull request for review.
