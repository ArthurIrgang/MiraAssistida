import cv2
import yaml
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

def undistortVideo(videoCapture, calibrationFile):
    
    # Load the calibration parameters from the YAML file
    with open(calibrationFile, 'r') as f:
        calib_data = yaml.safe_load(f)

        # Extract the camera matrix and distortion coefficients from the data
        camera_matrix = np.array(calib_data['camera_matrix'])
        dist_coeffs = np.array(calib_data['distortion_coefficients'])

    ret, frame = videoCapture.read()
    if not ret:
        # Exit the loop if there are no more frames to read
        return ret, frame
    
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newcameramtx)

    return ret, undistorted_img

def extractGreenRegion(videoCapture, lower_green=np.array([40, 40, 40]), upper_green=np.array([100, 255, 255]), kernel_size=(30,30)):
    
    # Read the next frame from the video stream
    ret, frame = undistortVideo(videoCapture,'dispersion_videos\calibration_parameters_14_3_2023.yaml')

    if not ret:
        w = 0
        h = 0
        # Exit the loop if there are no more frames to read
        return ret, frame, w, h

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
        green_screen_frame = frame[y:y+h, x:x+w]

        # Create a copy of the original frame
        original_frame = frame.copy()

        # Draw a rectangle around the green screen region on the original frame
        detection = cv2.rectangle(original_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #print(w,h) # width = 500 Pixels ; height = 332 pixels
        cv2.imshow("Detected Region", detection)

        return ret, green_screen_frame, w, h
    else: 
        print("Green screen not detected")

def ballTracking(videoCapture,imagesPath, frameDelay = 17):
    #videoCapture = cv2.VideoCapture('D:/UFRGS/TCC/MiraAssistida/testes/cropped_video.mp4')
    # dist = lambda x1,y1,x2,y2: ((x1-x2)**2 + (y1-y2)**2)/2
    drawLine = False
    # Create an empty list to store the center positions of the ball
    center_list = []
    frameNumber = 1
    
    while (True):
        
        ret, greenFrame, w, h = extractGreenRegion(videoCapture)

        if not ret:
            break
        width = w
        height  = h

        grayFrame = cv2.cvtColor(greenFrame, cv2.COLOR_BGR2GRAY)
        blurFrame = cv2.GaussianBlur(grayFrame, (5,5), 0)

        #greenFrame, x, y = detectBlackLine(greenFrame)   
        
        # imagem fonte, metodo, dp, distância mínima entre 2 círculos, sensibilidade, acurácia(numero de pontos de borda), raio minimo, raio maximo 
        circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 1, 1000,
                                    param1=90,param2=13,minRadius=1,maxRadius=15) 
    
    
        if circles is not None:
            circles = np.int32(np.around(circles))
            # print(circles, frameNumber)

            # Draw the center of the circle
            cv2.circle(greenFrame, (circles[0,0, 0], circles[0,0, 1]), 1, (0,100,100), 5)

            # Draw the edge of the circle
            cv2.circle(greenFrame, (circles[0,0, 0], circles[0,0, 1]), circles[0,0, 2], (255,0,255), 3)

            cv2.imwrite(imagesPath+"/frame%d.jpg" % frameNumber ,greenFrame)

            # Get the center position of the ball
            center = (circles[0,0,0], circles[0,0,1], circles[0,0,2], frameNumber)
            center_list.append(center)
        
        if drawLine:
            # Draw a red line between the center positions in the list to show the trajectory of the ball
            for i in range(1, len(center_list)):
                cv2.line(greenFrame, center_list[i-1], center_list[i], (0, 0, 255), thickness=2)

        # print(w,h)
        # Desenha as linhas dos eixos X e Y; segunda linha para EXP2
        # cv2.line(greenFrame, (0 , round(y))  , (w,round(y)) , (0, 0, 255), thickness=2)
        cv2.line(greenFrame, (0 , round(h/2))  , (w,round(h/2)) , (0, 0, 255), thickness=2)


        cv2.line(greenFrame, (round(w/2), 0)   , (round(w/2),h) , (0, 0, 255), thickness=2)


        # # Desenha linha corrigida com a matriz de rotação
        # cv2.line(greenFrame, (0                     , round((h/2)*1.0292)) , (round(w*1.0015)       ,round((h/2) * 1.0292))     , (255, 0, 255), thickness=2)
        # cv2.line(greenFrame, (round((w/2)*1.0015)   , 0)                    , (round((w/2)*1.0015)   ,round(h*1.0292))           , (255, 0, 255), thickness=2)

        # # Desenha linha corrigida com a matriz de rotação
        # cv2.line(greenFrame, (0                     , round((h/2)*0.97)) , (round(w*0.9979)       ,round((h/2) * 0.97))     , (255, 0, 0), thickness=2)
        # cv2.line(greenFrame, (round((w/2)*0.9979)   , 0)                    , (round((w/2)*0.9979)   ,round(h*0.97))           , (255, 0, 0), thickness=2)

 
        cv2.imshow("Circles", greenFrame)
        if cv2.waitKey(frameDelay) & 0xFF == ord('q'): 
            break

        frameNumber+=1

        

    # print('Total Number of Frames: ', frameNumber)
    # print("Lista de Centros nas Coordenadas da Imagem:\n", center_list)

    videoCapture.release()
    cv2.destroyAllWindows()
    # target_center = [[x,y]]
    target_center = [[round(w/2),round(h/2)]]
    frame_size = [width, height]
    
    return center_list, target_center, frame_size 

def detectBlackLine(frame, lower_black=np.array([0, 0, 0]), upper_black=np.array([180, 255, 30]), kernel_size=(100,5)):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Mask the green region in the image
    mask = cv2.inRange(hsv, lower_black, upper_black)

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
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        center_x = round((x+w+x)/2)
        center_y = round((y+h+y)/2)
        return frame, center_x, center_y
    else:
        return frame,
        
def extractFrames(sourceVideo, imagesPath): # Example of Path: 'dispersion_videos/source/4_Bar.mp4'
    # Opens the Video file
    cap = cv2.VideoCapture(sourceVideo)
    count=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imshow('window-name', frame)
        cv2.imwrite(imagesPath+"/frame%d.jpg" % count ,frame)
        count+=1
        if cv2.waitKey(17) & 0xFF == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()