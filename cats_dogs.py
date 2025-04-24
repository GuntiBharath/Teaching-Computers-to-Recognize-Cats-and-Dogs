import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from google.colab import files
from PIL import Image

# Load Pre-trained Model
model = MobileNetV2(weights="imagenet")

def classify_image(image_path):
    try:
        # Read and preprocess the image
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Resize to MobileNetV2 input size
        img_array = np.array(img)

        # Convert grayscale to RGB if needed
        if img_array.ndim == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Normalize

        # Predict
        predictions = model.predict(img_array)
        decoded_preds = decode_predictions(predictions, top=5)[0]  # Get top 5 predictions

        return decoded_preds, img  # Return predictions and image

    except Exception as e:
        print(f"Error: {e}")
        return None, None

# Upload an image
print("Please upload an image:")
uploaded = files.upload()

for filename in uploaded.keys():
    # Check for valid file type
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        print(f"Invalid file format: {filename}. Please upload a JPG or PNG.")
        continue

    # Classify uploaded image
    preds, img = classify_image(filename)

    if preds:
        # Display image
        plt.imshow(img)
        plt.axis("off")
        plt.show()

        # Print Predictions
        print("\nTop Predictions:")
        for i, (imagenet_id, label, confidence) in enumerate(preds):
            print(f"{i+1}: {label} (Confidence: {confidence:.2f})")
