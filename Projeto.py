import cv2
import mediapipe as mp

#inicialiar o modelo de deteccao de mãos do mediapipe
mp_maos = mp.solutions.hands #módulo com modelo pré treinado 
maos = mp_maos.Hands()
#utilitário para desenhar os pontos de referência da mão e suas conexões
mp_desenho = mp.solutions.drawing_utils


#cap recebe a captura de vídeo (o numero serve para selecionar qual das cameras será usada)
cap = cv2.VideoCapture(2);


while True:
    #ret e frame recebem informações, o ret recebe bool se ta dando vídeo, frame recebe o frame capturado pela camera
    ret, frame = cap.read()
    #se ret der falso, significa que tem problema, e para o loop
    if not ret: 
        break

    #converte BGR para RGB (BGR é o padrão do OpenCV, e RGB o que o MediaPipe espera)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Processa o frame no modelo de detecção de mão do MediaPipe, armazenando o resultado na variável "resultado"
    resultado = maos.process(rgb)

    #se o resultado tiver mais que uma mão com marcas (pontos para cada mão), roda o código abaixo
    if resultado.multi_hand_landmarks:
        #para cada ponto na mão, roda uma vez o loop
        for marcacao_mao in resultado.multi_hand_landmarks:
            #desenha os pontos de conexão no frame do vídeo (como referencia tem o frame, a mão e os pontos da mão)
            mp_desenho.draw_landmarks(frame, marcacao_mao, mp_maos.HAND_CONNECTIONS)

    #chama a função imshow, em aspas é o nome que ta em cima do programa, e frame o frame atual q ta no loop
    cv2.imshow("Video12312", frame)

    #verifica se a tecla "q" foi apertada, se sim ele sai do programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#desativa o uso da câmera
cap.release()
#fecha todas as telas abertas pelo cv
cv2.destryAllWindows()