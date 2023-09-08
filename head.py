import dlib
import cv2
import numpy as np
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # process the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
    # process the landmarks
        # calculate head tilt angle
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        eye_center = np.mean(np.concatenate((left_eye, right_eye), axis=0), axis=0)
        nose_tip = landmarks[30]
        dx = nose_tip[0] - eye_center[0]
        dy = nose_tip[1] - eye_center[1]
        head_tilt_angle = np.arctan2(dy, dx) * 180 / np.pi
        cv2.putText(frame, f"Head tilt angle: {head_tilt_angle:.1f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Head movements", frame)
        if cv2.waitKey(1) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

