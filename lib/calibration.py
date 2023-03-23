import numpy as np
import cv2
import glob
import os
import yaml
from lib import video

# Define the size of the checkerboard pattern(mm)

pattern_size = (9, 9)
square_size = 20.0

# pattern_size = (15,9)
# square_size = 47

def getImages(cap, num_images = 20):

    # Define the path to the output directory
    output_dir = "calibration_images/new_raw_images"

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set the image format to JPEG
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    # Loop through the desired number of images
    for i in range(num_images):
        # Wait for the user to press the 'p' key
        print(f"Press 'p' to capture image {i+1}/{num_images}...")
        while True:
            ret, frame = cap.read()
            cv2.imshow("Press 'p' to capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('p'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                exit()

        # Capture an image from the camera
        ret, img = cap.read()

        # Save the image to the output directory
        filename = os.path.join(output_dir, f"image_{i+1}.jpg")
        cv2.imwrite(filename, img, encode_params)

    # Release the camera
    cap.release()

    print(f"{num_images} images captured and saved to {output_dir}.")

def intrinsic(detected_dir = "calibration_images/detected_images", corrected_dir = "calibration_images/corected_images"):

    # Create the output directory if it doesn't exist
    if not os.path.exists(detected_dir):
        os.makedirs(detected_dir)

    # Create the output directory if it doesn't exist
    if not os.path.exists(corrected_dir):
        os.makedirs(corrected_dir)

    # Set the image format to JPEG
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    # Define the termination criteria for the calibration algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), ..., (5,8,0)
    object_points = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    object_points *= square_size

    # Arrays to store object points and image points from all the images.
    object_points_list = [] # 3d point in real world space
    image_points_list = [] # 2d points in image plane

    # Get the paths of all the calibration images
    images = glob.glob('calibration_images/raw_images/*.jpg')

    # Loop through all the calibration images and find the corners of the checkerboard pattern
    i=0

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Image", gray)
        cv2.waitKey(300)
        cv2.destroyAllWindows()

        # Find the corners of the checkerboard pattern
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

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
            filename = os.path.join(detected_dir, f"detected{i+1}.jpg")
            cv2.imwrite(filename, img, encode_params)
        
        i+=1

    cv2.destroyAllWindows()

    # Calibrate the camera using the Zhang method
    ret, mtx, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points_list, image_points_list, gray.shape[::-1], None, None)

    # Print the camera matrix and distortion coefficients
        #print("Camera matrix:\n", mtx)
        #print("Distortion coefficients:\n", dist_coeffs)
    

    # Save the calibration parameters to a YAML file
    with open('calibration_parameters.yaml', 'w') as file:
        yaml_data = {
            'camera_matrix': mtx.tolist(), 
            'distortion_coefficients': dist_coeffs.tolist()
        }
        yaml.dump(yaml_data, file)

    # Load an image and undistort it using the calibration parameters
    img = cv2.imread('calibration_images/new_raw_images/image_14.jpg')
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, mtx, dist_coeffs, None, newcameramtx)

    # Display the raw image and the undistorted image
    cv2.imshow('raw image', img)
    cv2.imshow('corrected image', undistorted_img)

    cv2.imwrite("undistorted.jpg" ,undistorted_img)
    cv2.imwrite("distorted.jpg" ,img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

def extrinsic(videoCapture, calibration_file = 'calibration_parameters.yaml'):
    
    # Load the calibration parameters from the YAML file
    with open(calibration_file, 'r') as f:
        calib_data = yaml.safe_load(f)

    # Extract the camera matrix and distortion coefficients from the data
    camera_matrix = np.array(calib_data['camera_matrix'])
    dist_coeffs = np.array(calib_data['distortion_coefficients'])
    print("\nMatriz da Câmera:")
    print(camera_matrix)
    print("\nCoeficientes de Distorção:")
    print(dist_coeffs)

    # Define the termination criteria for the calibration algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Define the object points for the checkerboard pattern
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
    objp *= square_size
    
    while True:
        ret, img = videoCapture.read()
        if not ret: 
            break
        
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
        gray = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Camera", undistorted_img)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

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
        cv2.drawChessboardCorners(undistorted_img, pattern_size, corners2, ret)
        cv2.imshow('Checkerboard Corners', undistorted_img)
        cv2.waitKey(0)

        with open('calibration_parameters.yaml', 'w') as file:
            yaml_data = {
                'camera_matrix': camera_matrix.tolist(), 
                'distortion_coefficients': dist_coeffs.tolist(),
                'Rotation matrix': R.tolist(),
                'Translation vectors': tvecs.tolist()
            }
            yaml.dump(yaml_data, file)

        # Calculate the distance to the calibration pattern
        print("Distance to pattern: ", tvecs[2], "mm")

        videoCapture.release()
        cv2.destroyAllWindows()

        return camera_matrix, dist_coeffs, R, tvecs


def undistort(frame, calibration_file):
    # Load the calibration parameters from the YAML file
    with open(calibration_file, 'r') as f:
        calib_data = yaml.safe_load(f)

    # Extract the camera matrix and distortion coefficients from the data
    camera_matrix = np.array(calib_data['camera_matrix'])
    dist_coeffs = np.array(calib_data['distortion_coefficients'])

    # Load an image and undistort it using the calibration parameters
    img = cv2.imread(frame)
    h, w = img.shape[:2]
    print(h,w)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)

    # Display the raw image and the undistorted image
    cv2.imshow('raw image', img)
    cv2.imshow('corrected image', undistorted_img)

    cv2.imwrite("undistorted.jpg" ,undistorted_img)
    cv2.imwrite("distorted.jpg" ,img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
