#O prensente arquivo é um script de testes.

#O prensente arquivo é um script de testes.

import cv2

# Create a VideoCapture object to capture video from the default camera (i.e. webcam)
video_capture = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not video_capture.isOpened():
    print("Could not open video device")
    exit()

# Loop through frames from the camera and display them
while True:
    # Read a frame from the camera
    ret, frame = video_capture.read()

    # Check if a frame was successfully captured
    if not ret:
        print("Could not read frame")
        break

    # Display the frame
    cv2.imshow('frame', frame)

    # Wait for user input to exit the loop (press 'q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the window
video_capture.release()
cv2.destroyAllWindows()
