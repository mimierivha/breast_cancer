import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np


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
st.title('Breast Cancer Histopathology Image Classification')
st.write('This app uses a pre-trained machine learning model to classify breast cancer images as either benign or malignant. Upload an image of a biopsy to get started.')

# Add a slider to adjust the prediction threshold
threshold = st.slider('Prediction threshold:', 0.0, 1.0, 0.5, 0.05)

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
        if prediction > threshold:
            st.write('The image is malignant')
        else :
            st.write('The image is benign')

        # Add a visual representation of the prediction
        prob_malignant = round(prediction*100,2)
        prob_benign = round((1-prediction)*100,2)
        st.write(f'Probability of malignant tumor: {prob_malignant}%')
        st.write(f'Probability of benign tumor: {prob_benign}%')
        st.bar_chart({'Malignant': prob_malignant, 'Benign': prob_benign})



        # Add a feedback mechanism
        st.write('Was the prediction accurate? Please leave your feedback below.')
        feedback = st.text_input('Feedback:')
        if st.button('Submit'):
            # Save the feedback to a file or database
            st.write('Thank you for your feedback!')

        # Add a description of the model and its accuracy
        st.write('The model used in this app was trained on a dataset ofbreast cancer histopathology images to classify the images as either benign or malignant. The model has an accuracy of X% on the test set. The model uses a convolutional neural network (CNN) architecture to extract features from the images and make a prediction. The CNN consists of X layers and was trained using the Adam optimizer with a learning rate of X.')

        # Add an explanation of how to interpret the prediction
        st.write(f'The prediction threshold is set to {threshold}. If the model predicts a probability of malignancy greater than this threshold, the prediction is "malignant". If the predicted probability is less than the threshold, the prediction is "benign".')

        # Add a brief conclusion to the app
        st.write('Thank you for using our breast cancer histopathology image classification app. If you have any feedback or questions, please feel free to contact us.')
