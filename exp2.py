import cv2
import dlib
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the pre-trained facial landmark detector from dlib
predictor_path = 'D:/Drowsiness_Detector-dlib--main/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Load the pre-trained model
model_path = 'D:/Drowsiness_Detector-dlib--main/drowsiness_detector.model'
model = joblib.load(model_path)

# Function to extract facial landmarks from an image
def extract_facial_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) > 0:
        landmarks = []
        shape = predictor(gray, faces[0])

        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            landmarks.append((x, y))

        return np.array(landmarks)
    else:
        return None

# Function to preprocess the facial landmarks
def preprocess_landmarks(landmarks):
    flattened_landmarks = landmarks.reshape(1, -1)
    scaler = StandardScaler()
    scaled_landmarks = scaler.fit_transform(flattened_landmarks)
    return scaled_landmarks

# Open the video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = video_capture.read()

    if ret:
        # Extract facial landmarks from the frame
        landmarks = extract_facial_landmarks(frame)

        if landmarks is not None:
            # Preprocess the landmarks
            preprocessed_landmarks = preprocess_landmarks(landmarks)

            # Make predictions using the pre-trained model
            prediction = model.predict(preprocessed_landmarks)

            # Convert prediction to human-readable label
            label = "Awake" if prediction == 0 else "Drowsy"

            # Display the result
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("Drowsiness Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()
cv2.destroyAllWindows()
