import numpy as np
from preprocessor import preprocess_face



def predict_age(face_img,age_model):
    preprocessed_face = preprocess_face(face_img)
    
    # Debug: print the preprocessed face
    print(f"Preprocessed face shape: {preprocessed_face.shape}")
    print(f"Preprocessed face: {preprocessed_face}")
    
    age_prediction = age_model.predict(preprocessed_face)
    
    # Debug: print the raw prediction output
    print("Raw age_prediction:", age_prediction)
    
    # Assuming the second output array is the one with age probabilities
    age_probabilities = age_prediction[1][0]
    
    # Compute the predicted age as a weighted sum of age categories
    age_categories = np.arange(len(age_probabilities))
    predicted_age = int(np.dot(age_categories, age_probabilities))
    
    return predicted_age