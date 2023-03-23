import yaml
import matplotlib.pyplot as plt
import numpy as np


def calculateImpactPoints(center_list, filePath):
    impact_points = []
    shot_centers = []
    prev_frame_num = 1
    shot_frame_diff = 20 # set the maximum number of frames between shots
    shot_id = 0
    for center in center_list:

        frame_num = center[3]

        if (((frame_num - prev_frame_num) > shot_frame_diff) and shot_centers):
        # if the frame difference is larger than the shot_frame_diff and we have shot centers, then
        # consider the previous centers as a new shot and add the smallest center to the impact_points list

            if len(shot_centers) > 3:  # add this condition to ignore shots with less than 4 detected centers
                smallest_center = min(shot_centers, key=lambda c: c[2])
                impact_points.append(smallest_center)
                print("\nShot ", shot_id,": ", shot_centers)
                print("Shot ",shot_id, "impact point: ", smallest_center)
                shot_id += 1
            shot_centers = []
        shot_centers.append(center)
        prev_frame_num = frame_num


    # add the last shot
    if shot_centers:
        if len(shot_centers) > 3:  # add this condition to ignore shots with less than 4 centers
            smallest_center = min(shot_centers, key=lambda c: c[2])
            impact_points.append(smallest_center)
            print("\nShot ", shot_id,": ", shot_centers)
            print("Shot ",shot_id, "impact point: ", smallest_center)


    with open(filePath+'/impact_points.yaml', 'w') as file:
        yaml_data = {
            'shot_centers': np.array(shot_centers).tolist(),
            'impact_points': np.array(impact_points).tolist()

        }
        yaml.dump(yaml_data, file)


    return np.array(impact_points)

def plotDispersion(impact_points, title):
    impact_x = impact_points[:,0]
    impact_y = impact_points[:,1]
    
    plt.plot(impact_x,impact_y, 'ro')
    plt.axis([-610,610,-410,410])
    plt.xlabel("x Axis [mm]")
    plt.ylabel("y Axis [mm]")
    plt.title(title)
    plt.grid(True)
    plt.show()

def calculateStatistics(impact_points, filePath):
    y_points = impact_points[:,1]

    mean = np.mean(y_points)
    std_dev = np.std(y_points)
    median = np.median(y_points)
    q1, q3 = np.percentile(y_points, [25, 75])
    iqr = q3 - q1

    with open(filePath+'/statistic.yaml', 'w') as file:
        yaml_data = {
            'mean': mean.tolist(),
            'standard_deviation':std_dev.tolist(),
            'median':median.tolist(),
            'Q1':q1.tolist(),
            'Q3':q3.tolist(),
            'InterQuartileRange':iqr.tolist()
        }
        yaml.dump(yaml_data, file)

def boxPlot(impact_points):
    y_points = impact_points[:,1]
    # Set up the plot
    fig, ax = plt.subplots()
    ax.boxplot(y_points, vert = True)
    # Add labels for key statistics
    mean = np.mean(y_points)
    median = np.median(y_points)
    q1, q3 = np.percentile(y_points, [25, 75])
    iqr = q3 - q1
    std_dev = np.std(y_points)
    ax.text(1.1, mean, 'Mean: {:.2f}'.format(mean), ha='left', va='center', fontsize=12)
    ax.text(1.5, median, 'Median: {:.2f}'.format(median), ha='left', va='center', fontsize=12)
    ax.text(0.6, q1-1.5*iqr, 'Q1: {:.2f}'.format(q1), ha='left', va='bottom', fontsize=12)
    ax.text(0.6, q3+1.5*iqr, 'Q3: {:.2f}'.format(q3), ha='left', va='top', fontsize=12)
    ax.text(1.4, q1-1.5*iqr, 'IQR: {:.2f}'.format(iqr), ha='left', va='bottom', fontsize=12)
    ax.text(0.6, mean+std_dev, 'Std Dev: {:.2f}'.format(std_dev), ha='left', va='top', fontsize=12)

    plt.show()

def barPlot(means, std_devs, pressures):   
    # Set up the plot
    fig, ax = plt.subplots()
    ax.bar(pressures, means, yerr=std_devs, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_xlabel('Pressão[Bar]')
    ax.set_ylabel('Coordenada y de impacto [mm]')
    ax.set_title('Média e Desvio Padrão')

    # Add text labels for the mean and standard deviation
    for i in range(len(pressures)):
        ax.text(pressures[i], means[i], 'Média: {:.2f}\nDesvio Padrão: {:.2f}'.format(means[i], std_devs[i]), ha='center', va='bottom', fontsize=8)

    # Show the plot
    plt.show()

def imageToTarget(impact_points):
    
    points = []
    u = np.array(impact_points)[:,0]
    v = np.array(impact_points)[:,1]
    # print(u,v)
    
    for i in range(len(u)):
        x = (250 - u[i]) * (1200/500)
        y = (166 - v[i]) * (807/332)
        point = ([x, y])
        points.append(point)
        
    return np.array(points)

def calculateError(point, reference):
    y = point[:,1]
    error = y-reference
    
    return error

def calculateNextPressure(limits, P, error):
    if error > 0: 
        limits[1] = P
    if error < 0:
        limits[0] = P
    
    new_P = (limits[0]+limits[1])/2

    return new_P, limits
