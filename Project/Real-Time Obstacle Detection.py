import cv2
import numpy as np
from tensorflow.keras.models import load_model
import serial

# Load trained model
model = load_model("obstacle_detection_model.h5")

# Labels
labels = {0: 'Human', 1: 'Car', 2: 'Obstacle'}

# Initialize Arduino
arduino = serial.Serial(port='COM3', baudrate=9600, timeout=.1)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    resized_frame = cv2.resize(frame, (64, 64))
    normalized_frame = resized_frame / 255.0
    reshaped_frame = np.reshape(normalized_frame, (1, 64, 64, 3))

    # Prediction
    predictions = model.predict(reshaped_frame)
    class_id = np.argmax(predictions)
    label = labels[class_id]

    # Send data to Arduino
    if label == 'Obstacle':
        arduino.write(b'1')  # Send signal to Arduino
    else:
        arduino.write(b'0')

    # Display frame with label
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Obstacle Detection', frame)

    # Break with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
