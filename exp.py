import tensorflow as tf
import cv2

# Load the pre-trained model
model_path = "D:/Drowsiness_Detector-dlib--main/drowsiness_detector.model"
model = tf.keras.models.load_model(model_path)

# Set up the camera
cap = cv2.VideoCapture(0)

# Define some helper functions for preprocessing and postprocessing the data
def preprocess_image(image):
    # Resize the image to match the input shape of the model
    resized_image = cv2.resize(image, (24, 24))
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # Reshape the image to match the input shape of the model
    input_image = gray_image.reshape((1, 24, 24, 1))
    # Normalize the pixel values
    input_image = input_image / 255.0
    return input_image

def postprocess_output(output):
    # Convert the output probabilities to a binary classification (0 or 1)
    prediction = 1 if output > 0.5 else 0
    return prediction

# Start the camera loop
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    # Preprocess the image
    input_image = preprocess_image(frame)
    # Run inference on the model
    output_prob = model.predict(input_image)[0][0]
    # Postprocess the output
    prediction = postprocess_output(output_prob)
    # Display the result on the screen
    if prediction == 1:
        cv2.putText(frame, "Drowsy", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Not drowsy", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Drowsiness Detection", frame)
    # Exit the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
