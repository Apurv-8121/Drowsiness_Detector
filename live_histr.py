import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the video file
cap = cv2.VideoCapture(0)


# Define the histogram bins
hist_bins = 256

# Loop through the video frames
while cap.isOpened():
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate the histogram of the grayscale image
    hist, bins = np.histogram(gray.ravel(), hist_bins, [0, hist_bins])
    
    # Plot the histogram
    plt.plot(bins[:-1], hist)
    plt.xlim([0, hist_bins])
    plt.ylim([0, gray.shape[0]*gray.shape[1]/10])  # Adjust y-axis scale
    plt.show()
    
    cv2.imshow('Video', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
