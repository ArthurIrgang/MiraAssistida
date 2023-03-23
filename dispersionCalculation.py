import cv2
import glob
import yaml
import matplotlib.pyplot as plt
import numpy as np
from lib import calibration, mat, video


videos = glob.glob('dispersion_videos\Exp2\source\*.mp4')

for file in videos:
    fname = file[-8:-4]
    bar = fname[0]
    plotTitle = 'Pontos de impacto para disparo com %s Bar' % bar
    frameDir = "dispersion_videos/Exp2/cut/" + fname

    videoCapture = cv2.VideoCapture(file)

    center_list, reference, frame_size = video.ballTracking(videoCapture, frameDir)
    impact_points = mat.calculateImpactPoints(center_list, frameDir)
        
    # mat.plotDispersion(impact_points, plotTitle)
    mat.calculateStatistics(impact_points, frameDir)


    