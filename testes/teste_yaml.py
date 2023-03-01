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
img = cv2.imread('D:/UFRGS/TCC/MiraAssistida/calibration_images/raw_images/image_2.jpg')

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