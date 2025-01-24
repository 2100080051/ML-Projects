import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.model_selection import train_test_split

# Set the image size and batch size
IMG_SIZE = 224
BATCH_SIZE = 32

# Load the dataset
train_dir = 'C:\\Users\\nabhi\\Downloads\\pythonProject2\\train'  # Replace with your train data path
val_dir = 'C:\\Users\\nabhi\\Downloads\\pythonProject2\\test'  # Replace with your validation data path

# Load images using Keras's image_dataset_from_directory function
train_dataset = image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_dataset = image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Define the base model (ResNet50) and include the top=False to exclude the last fully connected layer
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    weights='imagenet'
)

# Freeze the layers of the base model to prevent them from being trained
base_model.trainable = False

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')  # Change the number of classes to 7
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display the model summary
model.summary()

# Train the model
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset
)

# Save the trained model
model.save('emotion_detection_model.h5')

