import yaml
import cv2
import numpy as np

# Load the calibration parameters from the YAML file
with open('calibration_parameters.yaml', 'r') as f:
    calib_data = yaml.safe_load(f)

# Extract the camera matrix and distortion coefficients from the data
camera_matrix = np.array(calib_data['camera_matrix'])
dist_coeffs = np.array(calib_data['distortion_coefficients'])
print("\nMatriz da Câmera:")
print(camera_matrix)
print("\nCoeficientes de Distorção:")
print(dist_coeffs)

# Load an image to correct the distortion
img = cv2.imread('D:/UFRGS/TCC/MiraAssistida/calibration_images/raw_images/image_5.jpg')

# Get the size of the image
h, w = img.shape[:2]

# Use the camera matrix and distortion coefficients to undistort the image
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
img_undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

# Display the original and undistorted images side by side
cv2.imshow('Original', img)
cv2.imshow('Undistorted', img_undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()

#--------------------------------------------------------------------------------------------------------------------------------------------
# Define the termination criteria for the calibration algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Define the size of the checkerboard pattern
pattern_size = (9, 9)
square_size = 20.0

# Define the object points for the checkerboard pattern
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
objp *= square_size

# Load an image to perform extrinsic calibration on
#img = cv2.imread('D:/FRGS/TCC/MiraAssistida/calibration_images/raw_images/image_5.jpg')
gray = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2GRAY)

# Find the corners of the checkerboard pattern in the image
ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

# If the corners are found, perform extrinsic calibration
if ret:
    # Refine the corners to subpixel accuracy
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Calculate the rotation and translation vectors
    ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)

    # Print the rotation and translation vectors
    R, _ = cv2.Rodrigues(rvecs)
    print("Rotation matrix:\n", R)
    print("Translation vectors:\n", tvecs)
    cv2.drawChessboardCorners(img_undistorted, pattern_size, corners2, ret)
    cv2.imshow('Checkerboard Corners', img_undistorted)
    cv2.waitKey(0)
