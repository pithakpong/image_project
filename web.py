import streamlit as st
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from tensorflow.keras.applications.efficientnet import preprocess_input

# Define custom CSS styles
st.markdown(
    """
    <style>
    /* Add some custom styles for a better appearance */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
        font-size: 16px;
    }
    .st-c {
        color: #4CAF50;  /* Text color for captions */
    }
    .st-bkg {
        background-color: #f0f0f0;  /* Background color for the app */
    }
    </style>
    """,
    unsafe_allow_html=True
)
def preprocess(image,selected_option):
    preprocessed_image = np.array(image)
    preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
    preprocessed_image = cv2.resize(preprocessed_image,(224,224))
    if selected_option == "model_fromscatch":
        preprocessed_image = preprocessed_image / 255.0
        displayimg = np.array(preprocessed_image).astype(np.float32)
        return cv2.cvtColor(displayimg,cv2.COLOR_RGB2BGR),[preprocessed_image]
    preprocessed_image = preprocess_input(np.array(preprocessed_image).astype(np.float32))
    displayimg = preprocessed_image / 255.0
    return cv2.cvtColor(displayimg,cv2.COLOR_RGB2BGR),[preprocessed_image]
def predict(model,img):
    img = np.array(img)
    img = img.reshape(img.shape[0],224,224,3)
    return model.predict(img)
def face_detect(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for i, (x, y, w, h) in enumerate(faces):
        return img[y:y+h, x:x+w]
    return img
def main():
    st.title("Image Classification and Face Detection")
    st.markdown("Upload an image, select a model, and click 'Predict.'", unsafe_allow_html=True)

    model_efficient = tf.keras.models.load_model("model/EfficientNetB0.h5", compile=False)
    model_efficient.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_fromscratch = tf.keras.models.load_model("model/fromscatch.h5", compile=False)
    model_fromscratch.compile(loss=tf.keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    model_dict = {"model_efficient": model_efficient, "model_fromscatch": model_fromscratch}
    options = list(model_dict.keys())
    
    selected_option = st.selectbox("Select a model:", options)
    model = model_dict[selected_option]
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image_left = Image.open(uploaded_file)
        imgshow, imgpredict = preprocess(face_detect(image_left), selected_option)
        
        # Set the background color for the entire app
        st.markdown('<style>body { background-color: #f0f0f0; }</style>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.image(image_left, caption="Raw Image", use_column_width=True)
        
        with col2:
            st.image(imgshow, caption="Preprocessed Image", use_column_width=True)
        
        # Center-align the button horizontally
        with col3:
            if st.button("Predict"):
                result = predict(model, imgpredict)
                if result[0][0] > result[0][1]: 
                    st.success("Predict class: Fake")
                else: 
                    st.success("Predict class: Real")
                st.write(f"Fake Probability: {round(result[0][0] * 100, 2)}%")
                st.write(f"Real Probability: {round(result[0][1] * 100, 2)}%")

if __name__ == "__main__":
    main()