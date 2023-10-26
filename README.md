# Deepfake Detection Project

## Overview

This project is aimed at detecting deepfake images using two different models: EfficientNetB0 and a custom Convolutional Neural Network (CNN) trained from scratch. The primary goals of this project are to analyze the performance and accuracy of these models and to provide an interactive application for deepfake detection.

## Models

1. **EfficientNetB0:**
   - This model leverages the pre-trained EfficientNetB0 architecture for deepfake detection.

2. **Custom CNN Model:**
   - A CNN model developed from scratch, tailored for deepfake detection.

## Analysis

To assess the performance and accuracy of the models, the following metrics will be considered:
- Precision
- Recall
- F1 Score
- ROC-AUC

## Application

The deepfake detection application offers the following features:

- **Face Detection:**
  - Utilizes a face detection algorithm to locate and crop faces in uploaded images.

- **Preprocessing:**
  - Preprocesses images to prepare them for model input, which may include resizing, normalization, etc.

- **Model Selection:**
  - Users can choose between the EfficientNetB0 and the custom CNN model for prediction.

- **Image Visualization:**
  - Displays the original uploaded image and the preprocessed version for user reference.

- **Prediction:**
  - After uploading an image and selecting a model, click the "predict" button to initiate the detection process.

- **Results:**
  - The application returns a message indicating whether the uploaded image is classified as "fake" or "real" and provides the probability for each class.

## Usage

To use the application, follow these steps:

1. Clone this repository to your local machine.

2. Run the application using the following command:
```streamlit run web.py```
3. In the application interface:
- Upload an image.
- Select the desired model (EfficientNetB0 or custom CNN).
- Click the "predict" button.

4. View the prediction results, including the classification (fake or real) and the probability for each class.
