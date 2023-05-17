import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the saved model
model_path = 'C:/Users/dell/Desktop/mimi/dissertation.h5'
model = tf.keras.models.load_model(model_path)

# Define a function to preprocess the image and make a prediction
def predict(image):
    # Resize the image to a larger size
    resized_img = image.resize((256, 256))

    # Crop the image to the expected dimensions
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = (256 + 224) / 2
    bottom = (256 + 224) / 2
    cropped_img = resized_img.crop((left, top, right, bottom))

    # Convert the image to a NumPy array and normalize the pixel values
    image_array = np.array(cropped_img) / 255.

    # Flatten the image to a 1D array
    flattened_image = image_array.reshape((1, -1))

    # Reshape the flattened image to the expected shape
    expected_shape = (1, 175616)
    if flattened_image.shape[1] < expected_shape[1]:
        flattened_image = np.pad(flattened_image, [(0, 0), (0, expected_shape[1] - flattened_image.shape[1])])
    elif flattened_image.shape[1] > expected_shape[1]:
        flattened_image = flattened_image[:, :expected_shape[1]]
    
    # Make a prediction using the model
    prediction = model.predict(flattened_image)

    # Return the prediction as a probability score
    return prediction[0][0]
# Define the layout of the app
st.title('Breast Cancer Classification')
st.write('Upload an image of a breast to classify it as benign or malignant')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Check the dimensions of the image
    st.write(f"Image size: {image.size}")

    # Make a prediction when the user clicks the 'Predict' button
    if st.button('Predict'):
        # Preprocess the image and make a prediction
        prediction = predict(image)

        # Display the prediction
        if prediction >0.5:
            st.write('The image is malignant')
        else :
            st.write('The image is benign')
