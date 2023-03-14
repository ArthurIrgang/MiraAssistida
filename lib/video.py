import cv2
import numpy as np



def openLive(videoResolution = 720, fps = 60):

    #Defines the video proportion
    cameraProportion = 16/9

    # Open a live video
    videoCapture = cv2.VideoCapture(0)

    # Set the resolution of the camera to be the same used in the calibration
    videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, videoResolution)
    videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, round(cameraProportion*videoResolution))
    videoCapture.set(cv2.CAP_PROP_FPS, fps) 


    return videoCapture

def lowPassFilter(image, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.float32) / (kernel_size[0] * kernel_size[1])
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image


def extractGreenRegion(videoCapture, lower_green=np.array([40, 100, 100]), upper_green=np.array([100, 255, 255]), kernel_size=(10,10)):
    # Read the next frame from the video stream
    ret, frame = videoCapture.read()

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Mask the green region in the image
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Perform morphological operations on the mask to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find the contour of the green region
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Extract the green screen region from the frame
        green_screen_region = frame[y:y+h, x:x+w]

        # Create a copy of the original frame
        original_frame = frame.copy()

        # Draw a rectangle around the green screen region on the original frame
        detection = cv2.rectangle(original_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("Detected Region", detection)

        return green_screen_region
    else: 
        print("Green screen not detected")


    

   

def ballTracking(videoCapture, frameDelay = 17):
    #videoCapture = cv2.VideoCapture('D:/UFRGS/TCC/MiraAssistida/testes/cropped_video.mp4')

    # dist = lambda x1,y1,x2,y2: ((x1-x2)**2 + (y1-y2)**2)/2

    # Create an empty list to store the center positions of the ball
    center_list = []

    while True:
        ret, frame = videoCapture.read()
        if not ret: 
            break
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurFrame = cv2.GaussianBlur(grayFrame, (5,5), 0)

        # imagem fonte, metodo, dp, distância mínima entre 2 círculos, sensibilidade, acurácia(numero de pontos de borda), raio minimo, raio maximo 
        circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 1, 1000,
                                    param1=100,param2=15,minRadius=1,maxRadius=15) 
    
    
        if circles is not None:
            circles = np.int32(np.around(circles))
            print(circles)
        
            # Draw the center of the circle
            cv2.circle(frame, (circles[0,0, 0], circles[0,0, 1]), 1, (0,100,100), 5)

            # Draw the edge of the circle
            cv2.circle(frame, (circles[0,0, 0], circles[0,0, 1]), circles[0,0, 2], (255,0,255), 3)
        
            # prevCircle = chosen

            # Get the center position of the ball
            center = (circles[0,0,0], circles[0,0,1])
            center_list.append(center)

        # Draw a red line between the center positions in the list to show the trajectory of the ball
        for i in range(1, len(center_list)):
            cv2.line(frame, center_list[i-1], center_list[i], (0, 0, 255), thickness=2)
    
        cv2.imshow("Circles", frame)
        if cv2.waitKey(frameDelay) & 0xFF == ord('q'): 
            break

    print("Lista de Centros nas Coordenadas da Imagem:\n", center_list)

    videoCapture.release()

    cv2.destroyAllWindows()