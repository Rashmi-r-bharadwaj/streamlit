import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Define image properties
img_size = 256
channels = 3
num_classes = 2

# Define training parameters
batch_size = 32
epochs = 2

# Define paths to data
cancer_dir = r"G:\New folder\Cancerous"
non_cancer_dir = r"G:\New folder\Non Cancerous"

# Load data
cancer_images = []
non_cancer_images = []

for img_path in os.listdir(cancer_dir):
    img = cv2.imread(os.path.join(cancer_dir, img_path))
    img = cv2.resize(img, (img_size, img_size))
    cancer_images.append(img)

for img_path in os.listdir(non_cancer_dir):
    img = cv2.imread(os.path.join(non_cancer_dir, img_path))
    img = cv2.resize(img, (img_size, img_size))
    non_cancer_images.append(img)

# Create labels
cancer_labels = np.zeros(len(cancer_images))
non_cancer_labels = np.ones(len(non_cancer_images))

# Combine data and labels
X = np.array(cancer_images + non_cancer_images)
y = np.concatenate([cancer_labels, non_cancer_labels])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Convert labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define the CNN model
model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(img_size, img_size, channels)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation="relu"),
    keras.layers.Dense(units=num_classes, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

# Evaluate the model on the testing set
score = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")

threshold = 0.5

def preprocess_image(image):
    img = cv2.resize(image, (img_size, img_size))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img
import streamlit as st
# Define the Streamlit app
def main():
    st.title("Cancer Classification")

    # Allow the user to upload an image
    uploaded_file = st.file_uploader("Choose an image", type=["png"])

    if uploaded_file is not None:
        # Read the image file and preprocess it
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        preprocessed_image = preprocess_image(image)

        # Classify the image using the trained model
        prediction = model.predict(preprocessed_image)[0]
        if prediction[0] > prediction[1]:
            st.write("The image is cancerous.")
        elif prediction[0] < prediction[1]:
            st.write("The image is non-cancerous.")
        elif prediction[0] < threshold and prediction[1] < threshold:
            st.write("The image is neither cancerous nor non-cancerous.")
if __name__ == "__main__":
    main()