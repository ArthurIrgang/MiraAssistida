import cv2
from lib import video

videoCapture = video.openLive()

while True:
    # Read a frame from the camera
    ret, frame = videoCapture.read()

    # Check if a frame was successfully captured
    if not ret:
        print("Could not read frame")
        break

    # Display the frame
    cv2.imshow('Original frame', frame)

    # Extract and show the green video
    greenScreen = video.extractGreenRegion(videoCapture)
    cv2.imshow("Green Screen Extraction", greenScreen)

    # Check for user input to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
videoCapture.release()
cv2.destroyAllWindows()

