import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Path dataset
train_path = "dataset/train"
val_path = "dataset/validation"

# Image data generator
datagen = ImageDataGenerator(rescale=1.0/255)
train_data = datagen.flow_from_directory(train_path, target_size=(64, 64), batch_size=32, class_mode='categorical')
val_data = datagen.flow_from_directory(val_path, target_size=(64, 64), batch_size=32, class_mode='categorical')

# Model definisi
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_data.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(train_data, validation_data=val_data, epochs=10)

# Save model
model.save("obstacle_detection_model.h5")
print("Model saved successfully!")
