# pip install tensorflow keras

import os
import cv2
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("models/sign_model/keras_model.h5")  # Update with your model path

# Load class labels (only alphabets)
with open("models/sign_model/labels.txt", "r") as f:
    labels = {int(line.split()[0]): line.split()[1] for line in f.readlines()}

# Function to preprocess the image
def preprocess_image(img_path, img_size=224):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to classify all images in a folder (in number-wise order)
def classify_images_in_folder(folder_path, threshold=None):
    label_outputs = []

    # Get sorted list of filenames based on number in filename
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))  # Sort by numeric value

    for filename in image_files:
        img_path = os.path.join(folder_path, filename)
        img = preprocess_image(img_path)
        prediction = model.predict(img, verbose=0)
        class_index = np.argmax(prediction)
        confidence = prediction[0][class_index]

        if threshold is not None and confidence < threshold:
            label_outputs.append("UNKNOWN")
        else:
            label_outputs.append(labels[class_index])

    return label_outputs


# Example usage
folder_path = "data/signs"
predicted_labels = classify_images_in_folder(folder_path)

# Print the list of predicted labels
print(predicted_labels)

# Save the list of predicted labels to a file as a Python list
os.makedirs("outputs", exist_ok=True)  # Make sure output folder exists

with open("outputs/predicted_labels.txt", "w") as f:
    f.write(str(predicted_labels))  # Saves like ['D', 'O', 'G']

print("Predicted labels saved to outputs/predicted_labels.txt as a list.")


