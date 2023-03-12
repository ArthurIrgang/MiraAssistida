import cv2
import numpy as np

#videoCapture = cv2.VideoCapture(0)
videoCapture = cv2.VideoCapture('D:/UFRGS/TCC/MiraAssistida/testes/cropped_video.mp4')
prevCircle = None

dist = lambda x1,y1,x2,y2: ((x1-x2)**2 + (y1-y2)**2)/2

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
        #chosen = None
        #for i in circles[0,:]:
           # if chosen is None: chosen = i
           # if prevCircle is not None:
           #     if dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) <= dist( i[0], i[1] ,prevCircle[0], prevCircle[1]):
           #         chosen=i
        
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
    if cv2.waitKey(300) & 0xFF == ord('q'): 
        break

print("Lista de Centros nas Coordenadas da Imagem:\n", center_list)
videoCapture.release()
cv2.destroyAllWindows()