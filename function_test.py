import cv2
import glob
import yaml
import matplotlib.pyplot as plt
import numpy as np
from lib import calibration, mat, video



# live = video.openLive(720, 60)
#  # Cria o vetor de frames onde a bola foi detectada
# center_list = video.ballTracking(live, 'live_images')
# print(center_list)







file = 'dispersion_videos/source/4_Bar.mp4'
fname = file[-9:-4]
bar = fname[0]
plotTitle = 'Pontos de impacto para disparo com %s Bar' % bar
frameDir = "dispersion_videos/cut/" + fname

videoCapture = cv2.VideoCapture(file)
center_list, reference, frame_size = video.ballTracking(videoCapture, frameDir)
impact_points = mat.calculateImpactPoints(center_list, frameDir)
print("Imagem\n", impact_points)
impact_points = mat.imageToTarget(impact_points)
print("Alvo \n", impact_points)

# u = 210
# v = 222
# r = 3


# with open('dispersion_videos\calibration_parameters_14_3_2023.yaml', 'r') as file:
#         calib_data = yaml.safe_load(file)
        
#         A = np.array(calib_data['camera_matrix'])
#         R = np.array(calib_data['Rotation matrix'])
#         t = np.array(calib_data['Translation vectors'])


# r,_ = cv2.Rodrigues(R)
# r = (r*360)/(2*np.pi)
# print(r)


# s = 19/(2*r) # diametro[mm] / diametro[pixel]

# print(         (u-A[0,2])/A[0,0]           )

# uv = np.array([[u,v,1]], dtype=np.float32)
# uv = uv.T
# s_uv = np.multiply(s,uv)
# inv_A = np.linalg.inv(A)
# print('A = \n',A)
# print(np.linalg.norm(A))
# P = np.dot(inv_A,s_uv)
# print(P)
# P = np.subtract(P,t)
# inv_R = np.linalg.inv(R)
# P = np.dot(inv_R,P)






# E = np.concatenate((R,t), axis=1)

# #print('\nE = \n',E)
# H = np.dot(A,E)
# #print('\nH = \n',H)
# inv_H = np.linalg.pinv(H)

# xy = np.dot(inv_A, s_uv)

# P = np.dot(inv_H,xy)
# P = np.divide(P,P[3])

#print('\nP = \n',P)
