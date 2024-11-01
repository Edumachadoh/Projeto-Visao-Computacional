import cv2
import mediapipe as mp
import numpy as np
import math 

#inicialiar o modelo de deteccao de mãos do mediapipe
mp_maos = mp.solutions.hands #módulo com modelo pré treinado 
maos = mp_maos.Hands()
#utilitário para desenhar os pontos de referência da mão e suas conexões
mp_desenho = mp.solutions.drawing_utils



#cap recebe a captura de vídeo (o numero serve para selecionar qual das cameras será usada)
cap = cv2.VideoCapture(2)

#variavel para armazenar a posição anterior do dedo indicador
posicao_anterior = None

#determina altura e largura do quadro que vai ser desenhado
largura, altura = 1280, 720

#determinar tamanho da captura da camera com base na altura e largura pre determinadas
cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)

#mp.zeros cria uma matriz com pontos de valor RGB[0,0,0], que supostamente ficam atras do video da camera
#quando for pintado algo na tela esse ponto na matriz muda de valor, logo tem cor e aparece na camera
tela = np.zeros((altura, largura, 3), dtype=np.uint8)

#controla se está desenhando ou não
desenhando = False

#define o controle da borracha
modo_borracha = False

def calcular_distancia(ponto1, ponto2):
    return math.sqrt((ponto2[0] - ponto1[0]) ** 2 + (ponto2[1] - ponto1[1]) ** 2)

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
            #cx = int(marcacao_mao.landmark[8].x * largura)
            #cy = int(marcacao_mao.landmark[8].y * altura)

            #define como x ou y polegar ou indicador a posição de cada um, multiplicado pela largura ou altura para transformar de coordenada em pixel
            x_polegar = int(marcacao_mao.landmark[4].x * largura)
            y_polegar = int(marcacao_mao.landmark[4].y * altura)
            x_indicador = int(marcacao_mao.landmark[8].x * largura)
            y_indicador = int(marcacao_mao.landmark[8].y * altura)
            #o mesmo que os de cima, mas para o dedo do meio
            x_dedo_meio = int(marcacao_mao.landmark[12].x * largura)
            y_dedo_meio = int(marcacao_mao.landmark[12].y * altura)

            #define como "distancia" a distância entre o polegar e o indicador
            distancia = calcular_distancia((x_polegar, y_polegar), (x_indicador, y_indicador)) 

            #define como "distancia_borracha" a distancia entre o indicador e o dedo do meio
            distancia_borracha = calcular_distancia((x_indicador, y_indicador), (x_dedo_meio, y_dedo_meio))

            #define o limite da pinça (distancia entre os dedos), mas pode ser ajustado
            limite_pinca = 45

            #define o limite da borracha (distancia entre os dedos), ajustado
            limite_borracha = 60

            #condicional, se a distancia entre os dedos for menor que o limite, desenha, se for maior não desenha 
            if distancia < limite_pinca:
                desenhando = True
            #outra condicional, se a distancia entre os dedos for menor que o limite, ativa o modo borracha e desativa desenhando
            elif distancia_borracha < limite_borracha:
                modo_borracha = True
                desenhando = False
            else:
                #se nenhuma das condicionais for verdadeira, nem desenha nem apaga
                desenhando = False
                modo_borracha = False
                #define posicao_anterior como None para não conflitar com a ultima posicao a ser desenhada (linha reta de um ponto a outro)
                posicao_anterior = None

            #verifica se posicao anterior é nula, caso nao for ele desenha uma linha com base na posicao anterior
            #tela=onde esta matriz a ser desenhada, (cx, cy)=posicao do dedo indicador an tela, (255, 255, 255)= cor do desenho,5=espessura
            if desenhando: #se desenhando for True, desenha na tela
                if posicao_anterior:
                    cv2.line(tela, posicao_anterior, (x_indicador, y_indicador), (255, 255, 255), 5)
                posicao_anterior = (x_indicador, y_indicador)
            elif modo_borracha: #se o modo_borracha for true
                #chama uma função do openCV, que desenha um círculo, passando como parâmetro a tela (onde será desenhado), raio do círculo, cor do círculo (no caso é preta pois o fundo de "tela" é preto) e -1 para o círculo estar preenchido com a cor preta, logo vai apagar a pintura
                cv2.circle(tela, (x_indicador, y_indicador), 20, (0, 0, 0), -1) 
            
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