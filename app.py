import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load your trained model (replace with your model's path)
model = load_model("model_file_30epochs.h5")  # Replace with the actual model path

# Emotion labels starting from index 0
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

    st.write("### Upload an image to detect the emotion")
    st.write(
        "This project uses a Convolutional Neural Network (CNN) to classify emotions from facial expressions. The model has been trained to detect emotions like Angry, Happy, Sad, etc., from input images.")

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
        emotion_text = f"Predicted Emotion: {emotion_labels[emotion]}"
        st.write(emotion_text)

        # Display probabilities
        probabilities_percentage = [f"{round(prob * 100, 2)}%" for prob in probabilities]
        st.write(f"Probabilities: {probabilities_percentage}")

        # Display a bar chart for the predicted probabilities
        st.write("### Emotion Distribution")
        fig, ax = plt.subplots()
        ax.bar(emotion_labels, probabilities)
        ax.set_xlabel('Emotion')
        ax.set_ylabel('Probability')
        ax.set_title('Emotion Prediction Probabilities')
        st.pyplot(fig)


# Display Local Images by Path
def display_local_images():
    st.title("Project Demonstration Images")

    st.write("### Here are some examples of images used in this project:")

    # Replace with your image paths
    image_paths = [
        "C:\\Users\\nabhi\\Downloads\\pythonProject2\\1.png",
        "C:\\Users\\nabhi\\Downloads\\pythonProject2\\2.png",
        "C:\\Users\\nabhi\\Downloads\\pythonProject2\\training_validation_accuracy.png"
    ]

    for img_path in image_paths:
        if os.path.exists(img_path):
            st.image(img_path, caption=f"Example Image - {os.path.basename(img_path)}", use_column_width=True)
        else:
            st.warning(f"Image not found: {img_path}")


# Live Webcam Testing Function
def live_test():
    st.title("Emotion Detection - Live Webcam")

    st.write("### Use your webcam for real-time emotion detection.")
    st.write("Click the button below to start the webcam. The system will detect emotions from the live feed.")

    # Start webcam for live testing
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict emotion
        emotion, probabilities = predict_emotion(frame)
        emotion_text = f"Emotion: {emotion_labels[emotion]}"

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
    # Sidebar section
    st.sidebar.title("Choose Input Mode")
    mode = st.sidebar.radio("Select Mode", ["Upload Image", "Live Webcam", "Project Images"])

    # Displaying Project Information
    st.markdown("""
    ## About the Project
    This is an Emotion Detection project that uses a Convolutional Neural Network (CNN) to detect emotions from facial expressions. The model has been trained to recognize 7 different emotions: **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**.

    ### Model Overview
    The model used in this project is a Convolutional Neural Network (CNN). CNNs are widely used for image classification tasks, especially in object recognition and facial expression classification. The model consists of several layers:
    - **Convolutional Layers**: To extract features from input images.
    - **Pooling Layers**: To reduce spatial dimensions and retain important features.
    - **Fully Connected Layers**: To classify the features into distinct emotions.
    - **Softmax Activation**: For the final layer to give probabilities for each emotion class.

    ### Training Details
    - **Dataset**: The model has been trained on the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013), which consists of over **35,000 images** of human faces in various emotional states. The dataset includes labeled images of faces expressing **7 emotions**: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
    - **Training Images**: Over **30,000 images** were used for training the model.
    - **Validation and Testing**: Around **5,000 images** were used for validation and testing to evaluate the model's performance.

    ### Accuracy and Performance
    - **Accuracy Achieved**: The model achieved **83% accuracy** on the validation set.
    - **Loss**: The model's loss function is cross-entropy, which was optimized using **Adam optimizer**.
    - **Training Time**: The model was trained for **25 epochs**, each taking approximately 10 minutes on a GPU.

    ### Technologies Used
    - **Python**: Programming language used for the project.
    - **TensorFlow/Keras**: Deep learning framework used to build and train the CNN model.
    - **OpenCV**: Used for preprocessing images and capturing live video for emotion detection.
    - **Streamlit**: For creating the interactive web application.
    - **pyttsx3** (optional): For text-to-speech functionality, providing voice feedback based on the detected emotion.

    ## How to Use
    1. **Upload Image**: Upload an image from your device to detect the emotion of the person in the image.
    2. **Live Webcam**: Use your webcam for real-time emotion detection. Click the 'Start Live Test' button, and the app will display the predicted emotion.
    3. **Results**: The app will display the predicted emotion and the corresponding probabilities for each emotion.
    """)

    # Conditional rendering based on selected mode
    if mode == "Upload Image":
        upload_image()
    elif mode == "Live Webcam":
        live_test()
    elif mode == "Project Images":
        display_local_images()


if __name__ == "__main__":
    main()
