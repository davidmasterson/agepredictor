import os
import cv2
import numpy as np
from keras.api.models import load_model
from predictor import predict_age


# Specify the path to your HDF5 model file
age_model_path = 'weights.28-3.73.hdf5'

# Print statement for debugging
print(f"Attempting to load model from: {age_model_path}")

# Load the age prediction model
try:
    age_model = load_model(age_model_path, compile=False)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file '{age_model_path}' not found.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load the Haar cascade face detector
cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)


video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # Draw a rectangle around the faces and predict age
    for (x, y, w, h) in faces:
        face_img = frames[y:y+h, x:x+w]
        
        # Debug: Print the shape and type of face_img
        print(f"Face image shape: {face_img.shape}, type: {type(face_img)}")
        
        predicted_age = predict_age(face_img,age_model)
        
        # Debug: Print the predicted age
        print(f"Predicted age: {predicted_age}")
        
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frames, f'Age: {predicted_age}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    # Display the resulting frame
    cv2.imshow('Video', frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
video_capture.release()
cv2.destroyAllWindows()
