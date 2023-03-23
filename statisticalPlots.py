import glob
import yaml
import numpy as np
from lib import mat
import matplotlib.pyplot as plt

folders = glob.glob('dispersion_videos/Exp2/cut/*')
# folders = glob.glob('dispersion_videos/cut/*')

means = []
std_devs = []
pressures = [3,4,5,6,7,8,9]
n=0
data_points = 0
for folder in folders:

    with open(folder + '/impact_points.yaml', 'r') as f:
        impact_data = yaml.safe_load(f)

        # Extract the camera matrix and distortion coefficients from the data
        impact_points = np.array(impact_data['impact_points'])
    data_points = len(impact_points)
    print('Experimento com %s Bar de pressão' % pressures[n])
    print('Numero de Disparos: %s' % data_points)
    impact_points = mat.imageToTarget(impact_points)

    mat.plotDispersion(impact_points, 'Ponto de contato para %s Bar de pressão e 2,3 metros de distância' % pressures[n])
    
    mat.calculateStatistics(impact_points, folder)
    with open(folder + '/statistic.yaml', 'r') as f:
        impact_data = yaml.safe_load(f)

        # Extract the camera matrix and distortion coefficients from the data
        mean = impact_data['mean']
        std_dev = impact_data['standard_deviation']

    means.append(mean)
    std_devs.append(std_dev)
    n+=1


mat.barPlot(means, std_devs, pressures)

