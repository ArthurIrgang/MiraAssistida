import cv2
import yaml
from lib import calibration, video, mat


impact_points = []
limits = [0,9]
    
# Pede ao usuário para definir a pressão do primeiro tiro:
P = float(input("Digite a pressão utilizada para o tiro: "))

# Cria o vetor de frames onde a bola foi detectada
file = input("Digite o caminho do arquivo do video gravado: ")
videoCapture = cv2.VideoCapture(file)
center_list, reference, frame_size = video.ballTracking(videoCapture, 'live_images')

# Pega o desvio padrão da pressão utilizada
folder = 'dispersion_videos\cut\%s_Bar\statistic.yaml' % round(P)
with open(folder, 'r') as f:
    statistical_data = yaml.safe_load(f)
    std_dev = statistical_data['standard_deviation']
    std_dev = std_dev*807/frame_size[1]
    print("Desvio padrão para %s em mm" % round(P), std_dev) 

# Decide qual o ponto de impacto
impact_point = mat.calculateImpactPoints(center_list, '')
print("Ponto de impacto(UV): ", impact_point)
impact_point = mat.imageToTarget(impact_point)
print("Ponto de impacto(XY) ",impact_point)

# Ajustar a referência
reference = mat.imageToTarget(reference)
print(reference)

# calcula o erro
error = mat.calculateError(impact_point, reference[:,1])
print("Erro ", error) 
impact_points.append(impact_point[0])

while (abs(error[0]) > (50)):
    P, limits = mat.calculateNextPressure(limits, P, error[0])
    print("Pressão do próximo tiro: ", P)
    # Indica para o usuário que o tiro pode ser efetuado
    file = input("Digite o caminho do arquivo do video gravado: ")
    videoCapture = cv2.VideoCapture(file)
    center_list, reference, frame_size = video.ballTracking(videoCapture, 'live_images')
    
    # Pega o desvio padrão da pressão utilizado
    folder = 'dispersion_videos\cut\%s_Bar\statistic.yaml' % round(P)
    with open(folder, 'r') as f:
        statistical_data = yaml.safe_load(f)
        std_dev = statistical_data['standard_deviation']
        std_dev = std_dev*807/frame_size[1]
        print("Desvio padrão para %s em mm" % round(P), std_dev) 
    
    # Decide qual o ponto de impacto
    impact_point = mat.calculateImpactPoints(center_list, '')
    print("Ponto de impacto(UV): ", impact_point)
    impact_point = mat.imageToTarget(impact_point)
    print("Ponto de impacto(XY) ",impact_point)
    
    # calcula o erro
    error = mat.calculateError(impact_point, reference)
    
    # Ajustar a referência
    reference = mat.imageToTarget(reference)
    print(reference)
    
    # calcula o erro
    error = mat.calculateError(impact_point, reference[:,1])
    print("Erro ", error) 

    impact_points.append(impact_point[0])

cv2.destroyAllWindows()
print("Pontos de Impacto: ", impact_points)
