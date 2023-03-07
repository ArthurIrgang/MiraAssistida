import cv2
import numpy as np

# Load the video file
cap = cv2.VideoCapture('D:/UFRGS/TCC/MiraAssistida/testes/pingpong_video.mp4')

# Define color range for white ball
lower_white = np.array([0, 0, 200])
upper_white = np.array([255, 50, 255])

# Create an empty list to store the center positions of the ball
center_list = []

while cap.isOpened():
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a binary mask of the ball using the HSV range
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Apply morphological operations to remove noise and smooth the mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    
    # Find the contours of the ball in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter the contours based on area, aspect ratio, and solidity
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x,y,w,h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            if aspect_ratio > 0.8 and aspect_ratio < 1.2:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                solidity = float(area) / cv2.contourArea(cv2.convexHull(cnt))
                if solidity > 0.8:
                    # Draw a bounding box around the contour
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    
                    # Get the center position of the ball
                    center = (int(x + w/2), int(y + h/2))
                    center_list.append(center)
                    
    # Draw a circle at each center position in the list to show the trajectory of the ball
    for center in center_list:
        cv2.circle(frame, center, 3, (0,0,255), -1)
    
    # Display the output frame with the trajectory
    cv2.imshow('Pingpong Ball Tracker', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()