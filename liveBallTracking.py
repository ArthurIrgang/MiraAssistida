import cv2
import yaml
from lib import calibration, video, mat


impact_points = []
limits = [0,9]

# live = video.openLive(720, 30)

# calibration.extrinsic(live)

live = video.openLive(720, 60)

while True:
    
    # Pede ao usuário para definir a pressão do primeiro tiro:
    P = float(input("Digite a pressão utilizada para o tiro: "))
    # Pega o desvio padrão da pressão utilizada
    folder = 'dispersion_videos\cut\%s_Bar\statistic.yaml' % round(P)
    with open(folder, 'r') as f:
        statistical_data = yaml.safe_load(f)
        std_dev = statistical_data['standard_deviation']
        std_dev = std_dev*807/337
        print(std_dev) #px
    # Indica para o usuário que o tiro pode ser efetuado
    print("\n Efetue o disparo, após o reconhecimento feche a janela utilizando a tecla 'q'." )
    # Cria o vetor de frames onde a bola foi detectada
    center_list, reference, frame_size = video.ballTracking(live, 'live_images')
    # Decide qual o ponto de impacto
    impact_point = mat.calculateImpactPoints(center_list, '')
    print(impact_point)
    impact_point = mat.imageToTarget(impact_point)
    print(impact_point)
    # calcula o erro
    error = mat.calculateError(impact_point, 0)
    print(error) 
    impact_points.append(mat.imageToTarget(impact_point))

    while (error[0] > 50 ):
        P, limits = mat.calculateNextPressure(limits, P, error)
        # Pega o desvio padrão da pressão utilizado
        folder = 'dispersion_videos\cut\%s_Bar\statistic.yaml', round(P)
        with open(folder, 'r') as f:
            statistical_data = yaml.safe_load(f)
            std_dev = statistical_data['standard_deviation']
        # Indica para o usuário que o tiro pode ser efetuado
        print("\n Efetue o disparo, após o reconhecimento feche a janela utilizando a tecla 'q'." )
        # Cria o vetor de frames onde a bola foi detectada
        center_list = video.ballTracking(live, 'live_images')
        # Decide qual o ponto de impacto
        impact_point = mat.calculateImpactPoints(center_list, '')
        # calcula o erro
        error = mat.calculateError(impact_point, 0)
        impact_points.append(mat.imageToTarget(impact_point))

    cv2.destroyAllWindows()
