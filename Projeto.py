import cv2
import mediapipe as mp
import numpy as np

#inicialiar o modelo de deteccao de mãos do mediapipe
mp_maos = mp.solutions.hands #módulo com modelo pré treinado 
maos = mp_maos.Hands()
#utilitário para desenhar os pontos de referência da mão e suas conexões
mp_desenho = mp.solutions.drawing_utils



#cap recebe a captura de vídeo (o numero serve para selecionar qual das cameras será usada)
cap = cv2.VideoCapture(0)

#variavel para armazenar a posição anterior do dedo indicador
posicao_anterior = None

#determina altura e largura do quadro que vai ser desenhado
largura, altura = 640, 480

#determinar tamanho da captura da camera com base na altura e largura pre determinadas
cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)

#mp.zeros cria uma matriz com pontos de valor RGB[0,0,0], que supostamente ficam atras do video da camera
#quando for pintado algo na tela esse ponto na matriz muda de valor, logo tem cor e aparece na camera
tela = np.zeros((altura, largura, 3), dtype=np.uint8)


while True:
    #ret e frame recebem informações, o ret recebe bool se ta dando vídeo, frame recebe o frame capturado pela camera
    ret, frame = cap.read()
    #se ret der falso, significa que tem problema, e para o loop
    if not ret: 
        break

    frame = cv2.flip(frame, 1)

    #converte BGR para RGB (BGR é o padrão do OpenCV, e RGB o que o MediaPipe espera)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Processa o frame no modelo de detecção de mão do MediaPipe, armazenando o resultado na variável "resultado"
    resultado = maos.process(rgb)

    #se o resultado tiver mais que uma mão com marcas (pontos para cada mão), roda o código abaixo
    if resultado.multi_hand_landmarks:
        #para cada ponto na mão, roda uma vez o loop
        for marcacao_mao in resultado.multi_hand_landmarks:
            #obter coordenadas x e y do indicador
            cx = int(marcacao_mao.landmark[8].x * largura)
            cy = int(marcacao_mao.landmark[8].y * altura)
            
            #verifica se posicao anterior é nula, caso nao for ele desenha uma linha com base na posicao anterior
            #tela=onde esta matriz a ser desenhada, (cx, cy)=posicao do dedo indicador an tela, (255, 255, 255)= cor do desenho,5=espessura
            if posicao_anterior:
                cv2.line(tela, posicao_anterior, (cx, cy), (255, 255, 255), 5)
            posicao_anterior = (cx, cy)
            
            #desenha os pontos de conexão no frame do vídeo (como referencia tem o frame, a mão e os pontos da mão)
            mp_desenho.draw_landmarks(frame, marcacao_mao, mp_maos.HAND_CONNECTIONS)

    #combinar tela da camera com tela que foi feito o desenho
    frame_com_desenho = cv2.add(frame, tela)
    
    #chama a função imshow, em aspas é o nome que ta em cima do programa, e frame o frame atual q ta no loop, porem agora é desenho + camera
    cv2.imshow("Video12312", frame_com_desenho)

    #verifica se a tecla "q" foi apertada, se sim ele sai do programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#desativa o uso da câmera
cap.release()
#fecha todas as telas abertas pelo cv
cv2.destroyAllWindows()