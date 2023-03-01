import cv2
import glob


# Get the paths of all the calibration images
images = glob.glob('calibration_images/*.jpg')

# Loop through all the calibration images and find the corners of the checkerboard pattern
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Image", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
