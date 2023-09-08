# Drowsiness_Detector
This project aims at designing a smart and automated sleep detection system to alert the driver and 
wake him awake by buzzing an alarm within a fraction of a second after detecting the sleep.
1. **Drowsiness Detection:**
   - It uses a webcam feed to detect drowsiness in a person's eyes.
   - It calculates the eye aspect ratio and sounds an alarm if the aspect ratio falls below a certain threshold for a consecutive number of frames.
   - Additionally, it uses the OpenCV Haar Cascade Classifier to draw rectangles around detected faces.
   - It uses the dlib library for face detection and facial landmark prediction to calculate the eye aspect ratio.

2. **Yawn Detection:**
   - It uses a webcam feed to detect if a person is yawning.
   - It uses the dlib library to detect facial landmarks and calculates the mouth aspect ratio.
   - If the mouth aspect ratio exceeds a certain threshold, it indicates a yawn, and a message is displayed.

Here's a summary of each part:

**Drowsiness Detection:**
- It imports necessary libraries, including OpenCV, dlib, and Pygame (for playing a sound alert).
- It sets thresholds and variables for drowsiness detection, such as `EYE_ASPECT_RATIO_THRESHOLD` and `EYE_ASPECT_RATIO_CONSEC_FRAMES`.
- It loads a Haar Cascade Classifier for face detection.
- It initializes the dlib face detector and facial landmark predictor.
- It enters a loop to continuously capture frames from the webcam.
- For each frame:
  - It detects faces using the Haar Cascade Classifier and draws rectangles around them.
  - It uses dlib to detect facial landmarks, specifically the left and right eyes.
  - It calculates the eye aspect ratio and checks if it falls below the threshold for a certain number of consecutive frames.
  - If drowsiness is detected, it plays a sound alert and displays a message.
  - If the user is not drowsy, it stops the sound alert.

**Yawn Detection:**
- It defines functions to detect yawning using facial landmarks.
- It uses dlib to detect facial landmarks and extract the mouth region.
- It calculates the mouth aspect ratio.
- If the mouth aspect ratio exceeds a threshold, it indicates a yawn, and a message is displayed.
- This part is currently commented out (`#if cv2.waitKey(1) & 0xFF == ord('q'): break`) at the end of the loop, so it doesn't exit the program when 'q' is pressed.

It's important to note that these two parts are separate and not combined into a single script. You can use them individually for drowsiness detection and yawn detection, or you can integrate them into a more comprehensive driver monitoring system. Additionally, make sure to adjust the threshold values and fine-tune the code for specific use cases and desired sensitivities.
