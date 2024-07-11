import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load your pre-trained model
model_path = 'D:\Glaucoma streamlit\Glaucoma-classification-streamlit-app\glaucoma_classification_model.h5'
model = tf.keras.models.load_model(model_path)

# Define a function to make predictions
def predict_image(image):
    try:
        # Resize the image to match the input shape expected by the model
        image = image.resize((128, 128))  # Resize to 128x128
        image = image.convert('L')  # Convert to grayscale
        image = np.array(image)
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize the image to [0, 1] range

        # Make predictions
        predictions = model.predict(image)
        return predictions

    except Exception as e:
        st.write(f"Error during prediction: {str(e)}")
        return None

# Streamlit app
st.title("Image Classification App")

uploaded_file = st.file_uploader("Select An Image From the Folder:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying The Selected Image")

    # Predict the image
    predictions = predict_image(image)

    if predictions is not None:
        # Assuming your model output is a probability distribution over classes
        class_names = ['glaucoma', 'normal']  # Replace with your actual class names
        predicted_class = class_names[np.argmax(predictions)]

        st.write(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {np.max(predictions) * 100:.2f}%")

