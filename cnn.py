import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Fix the paths to data directories
train_data_dir = 'C:\\Users\\nabhi\\Downloads\\pythonProject2\\train'
validation_dir = 'C:\\Users\\nabhi\\Downloads\\pythonProject2\\test'

# Initialize counters for images
num_train_img = 0
for root, dirs, files in os.walk(train_data_dir):
    num_train_img += len(files)

num_validation_img = 0
for root, dirs, files in os.walk(validation_dir):
    num_validation_img += len(files)

print(num_train_img, num_validation_img)

# Model building section
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Data augmentation and generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Define class labels
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

# Model training with callbacks (EarlyStopping and ModelCheckpoint)
epochs = 30
history = model.fit(
    train_generator,
    steps_per_epoch=num_train_img // 32,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_validation_img // 32
)

# Save the model
model.save('cnn.h5')

# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))

# Plot training accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Check if validation accuracy exceeds 75%
final_accuracy = history.history['val_accuracy'][-1]  # Final validation accuracy
if final_accuracy > 0.75:
    print(f"Model Successful with validation accuracy: {final_accuracy*100:.2f}%")
else:
    print(f"Model did not achieve 75% accuracy. Final validation accuracy: {final_accuracy*100:.2f}%")
