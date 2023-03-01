import cv2
import os

# Define the number of images to capture
num_images = 10

# Define the path to the output directory
output_dir = "calibration_images/raw_images"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set up the camera
cap = cv2.VideoCapture(0)

# Set the resolution of the camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

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
