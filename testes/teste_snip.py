import cv2
import numpy as np

# Read video file
cap = cv2.VideoCapture('D:/UFRGS/TCC/MiraAssistida/testes/pingpong_video.mp4')

# Define color range for green background
lower_green = np.array([30, 50, 50])
upper_green = np.array([90, 255, 255])

# Create empty list to store cropped frames
cropped_frames = []

# Loop through each frame in the video
while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold frame to only include green pixels
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours of green background
    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through each contour and find the one with the largest area
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Crop frame to area of green background
    if max_contour is not None:
        # Get bounding box of contour
        x, y, w, h = cv2.boundingRect(max_contour)

        # Crop frame to bounding box
        cropped_frame = frame[y:y+h, x:x+w]

        # Add cropped frame to list
        cropped_frames.append(cropped_frame)

    # Display frame with green background detected
    cv2.imshow('frame', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

# Write cropped frames to new video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('cropped_video.mp4', fourcc, 30.0, (w, h))

for frame in cropped_frames:
    out.write(frame)

# Release video writer
out.release()
