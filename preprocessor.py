import numpy as np
import cv2


def preprocess_face(face_img):
    # Resize the input image to (64, 64)
    resized_face = cv2.resize(face_img, (64, 64))
    # Normalize pixel values to [0, 1]
    normalized_face = resized_face.astype(np.float32) / 255.0
    # Add batch dimension (expand_dims) to match model input shape
    processed_face = np.expand_dims(normalized_face, axis=0)
    return processed_face