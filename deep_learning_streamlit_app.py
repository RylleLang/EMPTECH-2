import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image, ImageOps
import os

MODEL_PATH = "mnist_cnn_model.h5"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def train_and_save_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Expand dims to add channel dimension
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Build a simple CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    # Save the model
    model.save(MODEL_PATH)

def preprocess_image(image: Image.Image):
    # Convert to grayscale
    image = ImageOps.grayscale(image)
    # Resize to 28x28
    image = image.resize((28,28))
    # Invert colors (Streamlit white background is white, MNIST digits are white on black)
    image = ImageOps.invert(image)
    # Convert to numpy array and normalize
    img_array = np.array(image).astype('float32') / 255.0
    # Expand dims to match model input
    img_array = np.expand_dims(img_array, axis=(0,-1))
    return img_array

def main():
    st.title("MNIST Digit Recognition with Deep Learning")
    st.write("This app trains a CNN on the MNIST dataset and allows you to draw a digit or upload an image for prediction.")

    if not os.path.exists(MODEL_PATH):
        st.write("Training model, please wait...")
        train_and_save_model()
        st.write("Model trained and saved!")

    model = load_model()

    st.write("Draw a digit or upload an image for prediction:")

    # Drawing canvas alternative: use file uploader for simplicity
    uploaded_file = st.file_uploader("Upload an image of a digit (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        st.write(f"Predicted Digit: {predicted_digit}")

    else:
        st.write("Or draw a digit on a white background and upload it.")

