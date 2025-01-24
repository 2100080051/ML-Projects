import cv2
import numpy as np
import streamlit as st
from keras.models import load_model

# Load your trained model (replace with your model's path)
model = load_model("model_file_30epochs.h5")  # Replace with the actual model path
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Neutral", "Surprise"]


# Preprocessing Function
def preprocess_image(image):

    resized_image = cv2.resize(image, (48, 48))
    if len(resized_image.shape) == 3:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    img_array = resized_image / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Prediction Function
def predict_emotion(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    emotion = np.argmax(predictions)
    probabilities = predictions[0]
    return emotion, probabilities


# Streamlit UI for Image Upload
def upload_image():
    st.title("Emotion Detection - Upload Image")

    # Upload Image
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert uploaded file to image
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict emotion
        emotion, probabilities = predict_emotion(image)
        emotion_text = f"Predicted Emotion: {emotion}"
        st.write(emotion_text)
        st.write(f"Probabilities: {probabilities}")


# Live Webcam Testing Function
def live_test():
    st.title("Emotion Detection - Live Webcam")

    # Start webcam for live testing
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict emotion
        emotion, probabilities = predict_emotion(frame)
        emotion_text = f"Emotion: {emotion}"

        # Display emotion on frame
        cv2.putText(frame, emotion_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert frame for Streamlit display
        st.image(frame, channels="BGR", caption="Emotion Detection", use_column_width=True)

        # Break loop if user clicks 'Stop Live Test'
        if st.button('Stop Live Test'):
            break

    cap.release()


# Streamlit main function to switch between upload and live testing
def main():
    st.sidebar.title("Choose Input Mode")
    mode = st.sidebar.radio("Select Mode", ["Upload Image", "Live Webcam"])

    if mode == "Upload Image":
        upload_image()
    elif mode == "Live Webcam":
        live_test()


if __name__ == "__main__":
    main()
