import cv2 as cv
import mediapipe as mp
import time
import utils, math
import numpy as np

# declaracao de variaveis
contador_frame =0
contador_direita=0
contador_esquerda =0
contador_centro =0 
FONTS = cv.FONT_HERSHEY_COMPLEX

# identificacao dos pontos de referencia
FACE_OVAL           = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]
LABIO               = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LABIO_INFERIOR      = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LABIO_SUPERIOR      = [185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
OLHO_ESQUERDO       = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
SOBRANC_ESQUERDA    = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
OLHO_DIREITO        = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]  
SOBRANC_DIREITA     = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

mapa_face_mesh = mp.solutions.face_mesh

# identificacao da camera
camera = cv.VideoCapture(0)
_, frame = camera.read()
img = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
img_altura, img_largura = img.shape[:2]
print(img_altura, img_largura)

# funcao de identificacao dos pontos (landmark detection) 
def landmarksDetection(img, resultados, draw=False):
    img_altura, img_largura= img.shape[:2]
    mesh_coord = [(int(point.x * img_largura), int(point.y * img_altura)) for point in resultados.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # retorna a lista de tuplas para cada landmark
    return mesh_coord

# Função de extratação de olhos
def extracaoOlhos(img, olho_direito_coords, olho_esquerdo_coords):
    # converte imagem para escala de cinza
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # obtem a dimensão da imagem
    dim = gray.shape

    # cria máscara a partir da dimensão da imagem em escala de cinza
    mask = np.zeros(dim, dtype=np.uint8)

    # desenha a forma dos olhos como uma máscara de cor branca
    cv.fillPoly(mask, [np.array(olho_direito_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(olho_esquerdo_coords, dtype=np.int32)], 255)

    # exibe a máscara 
    # cv.imshow('mask', mask)
    
    # desenha os olhos dentro da forma da mascara
    olhos = cv.bitwise_and(gray, gray, mask=mask)
    # muda a cor preta para cinza 
    # cv.imshow('olhos draw', olhos)
    olhos[mask==0]=155
    
    # obtem (x,y) mínimos e máximos para os olhos direito e esquerdo 
    # olho direito
    r_max_x = (max(olho_direito_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(olho_direito_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(olho_direito_coords, key=lambda item : item[1]))[1]
    r_min_y = (min(olho_direito_coords, key=lambda item: item[1]))[1]

    # olho esquerdo
    l_max_x = (max(olho_esquerdo_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(olho_esquerdo_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(olho_esquerdo_coords, key=lambda item : item[1]))[1]
    l_min_y = (min(olho_esquerdo_coords, key=lambda item: item[1]))[1]

    # recorta os olhos da máscara
    recorte_direito = olhos[r_min_y: r_max_y, r_min_x: r_max_x]
    recorte_esquerdo = olhos[l_min_y: l_max_y, l_min_x: l_max_x]

    # retorna os olhos recortados 
    return recorte_direito, recorte_esquerdo

# Estimador de posição dos olhos
def estimadorPosicao(olhos_recortados):
    # obtem altura e largura dos olhos 
    h, w = olhos_recortados.shape
    
    # remove o ruido das imagens
    gaussain_blur = cv.GaussianBlur(olhos_recortados, (9,9),0)
    median_blur = cv.medianBlur(gaussain_blur, 3)

    # aplica thrsholding para converter  para imagem binária 
    ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)

    # cria um "pedaco" da imagem binária
    pedaco = int(w/3) 

    # divide a imagem binária em 3 "pedaços"
    # cada pedaço representa uma parte do olho
    parte_direita = threshed_eye[0:h, 0:pedaco]
    parte_centro = threshed_eye[0:h, pedaco: pedaco+pedaco]
    parte_esquerda = threshed_eye[0:h, pedaco +pedaco:w]
    
    # chama a funcao para contar os pixels de cada pedaço
    posicao_olhos, color = pixelCounter(parte_direita, parte_centro, parte_esquerda)

    return posicao_olhos, color 

# define a funcao de contagem de pixels 
def pixelCounter(first_pedaco, second_pedaco, third_pedaco):
    # contagem de pixels pretos em cada pedaço 
    parte_direita = np.sum(first_pedaco==0)
    parte_central = np.sum(second_pedaco==0)
    parte_esquerda = np.sum(third_pedaco==0)
    # cria uma lista com esses valores
    partes_olhos = [parte_direita, parte_central, parte_esquerda]

    # obtem os maiores índices na lista 
    max_index = partes_olhos.index(max(partes_olhos))
    posicao_olho ='' 
    if max_index==0:
        posicao_olho="DIREITA"
        color=[utils.BLACK, utils.GREEN]
    elif max_index==1:
        posicao_olho = 'CENTRO'
        color = [utils.YELLOW, utils.PINK]
    elif max_index ==2:
        posicao_olho = 'ESQUERDA'
        color = [utils.GRAY, utils.YELLOW]
    else:
        posicao_olho="FECHADO"
        color = [utils.GRAY, utils.YELLOW]
    return posicao_olho, color


with mapa_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5, refine_landmarks=True) as face_mesh:

    start_time = time.time()
    
    while True:
        contador_frame +=1 
        ret, frame = camera.read() 
        if not ret: 
            break 
        
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_altura, frame_largura= frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        resultados  = face_mesh.process(rgb_frame)

        if resultados.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, resultados, False)
        
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in OLHO_ESQUERDO ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in OLHO_DIREITO], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
        
            right_coords = [mesh_coords[p] for p in OLHO_DIREITO]
            left_coords = [mesh_coords[p] for p in OLHO_ESQUERDO]

            recorte_direita, recorte_esquerda = extracaoOlhos(frame, right_coords, left_coords)
            # cv.imshow('right', recorte_direita)
            # cv.imshow('left', recorte_esquerda)

            posicao_olhos_direita, color = estimadorPosicao(recorte_direita)
            posicao_olhos_esquerda, color = estimadorPosicao(recorte_esquerda)
            
            if posicao_olhos_direita=="DIREITA" and contador_direita<2:
                start = time.process_time()
                contador_direita+=1
                contador_centro=0
                contador_esquerda=0
 
                end = time.process_time()
                print("Direita:", end - start)

            if posicao_olhos_direita=="CENTRO" and contador_centro <2:
                start = time.process_time()
                contador_centro +=1
                contador_direita=0
                contador_esquerda=0
                
                end = time.process_time()
                print("Centro:", end - start)

            
            if posicao_olhos_direita=="ESQUERDA" and contador_esquerda<2: 
                contador_esquerda +=1
                start = time.process_time()
                contador_centro=0
                contador_direita=0
                end = time.process_time()
                print("Esquerda:", end - start)

        end_time = time.time()-start_time
        fps = contador_frame/end_time
            
        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        if key==ord('q') or key ==ord('Q'):
            break
    cv.destroyAllWindows()
    camera.release()
