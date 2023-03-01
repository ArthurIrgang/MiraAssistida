import numpy as np
import cv2
import glob
import os
import yaml


detected_dir = "calibration_images/detected_images"
corrected_dir = "calibration_images/corected_images"

# Create the output directory if it doesn't exist
if not os.path.exists(detected_dir):
    os.makedirs(detected_dir)

# Create the output directory if it doesn't exist
if not os.path.exists(corrected_dir):
    os.makedirs(corrected_dir)

# Set the image format to JPEG
encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

# Define the size of the checkerboard pattern
pattern_size = (9, 13)

# Define the size of the squares in the checkerboard pattern in mm
square_size = 19.0

# Define the termination criteria for the calibration algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), ..., (5,8,0)
object_points = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
object_points *= square_size

# Arrays to store object points and image points from all the images.
object_points_list = [] # 3d point in real world space
image_points_list = [] # 2d points in image plane.

# Get the paths of all the calibration images
images = glob.glob('calibration_images/raw_images/*.jpg')

# Loop through all the calibration images and find the corners of the checkerboard pattern
i=0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Image", gray)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

    # Find the corners of the checkerboard pattern
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)

    # If the corners are found, add object points and image points to the lists
    if ret:
        object_points_list.append(object_points)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        image_points_list.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
        cv2.imshow('Checkerboard Corners', img)
        cv2.waitKey(500)
        # Save the image to the output directory
        filename = os.path.join(detected_dir, f"detected.jpg")
        cv2.imwrite(filename, img, encode_params)
        i+=1
    


cv2.destroyAllWindows()

# Calibrate the camera using the Zhang method
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points_list, image_points_list, gray.shape[::-1], None, None)

# Print the camera matrix and distortion coefficients
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)
# Save the calibration parameters to a YAML file
with open('calibration_parameters.yaml', 'w') as file:
    yaml_data = {'camera_matrix': mtx.tolist(), 'distortion_coefficients': dist.tolist()}
    yaml.dump(yaml_data, file)

print("Calibration complete. Calibration parameters saved to 'calibration_parameters.yaml'.")

# Load an image and undistort it using the calibration parameters
img = cv2.imread('calibration_images/raw_images/image_6.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)

# Display the raw image and the undistorted image
cv2.imshow('raw image', img)
cv2.imshow('corrected image', undistorted_img)
cv2.waitKey(0)

cv2.destroyAllWindows()
