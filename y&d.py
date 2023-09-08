import cv2
import dlib
import imutils
from imutils import face_utils
import numpy as np

def detect_yawn(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)
        if mar > 0.6:
            cv2.putText(frame, "Yawn detected!", (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return True
    return False

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[3] - mouth[9])
    B = np.linalg.norm(mouth[2] - mouth[10])
    C = np.linalg.norm(mouth[4] - mouth[8])
    D = np.linalg.norm(mouth[0] - mouth[6])
    mar = (A + B + C) / (3.0 * D)
    return mar

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:/Drowsiness_Detector-dlib--main/shape_predictor_68_face_landmarks.dat')
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    if detect_yawn(frame):
        print("Yawn detected!")
    cv2.imshow('frame',frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
