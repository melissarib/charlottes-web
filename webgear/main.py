import uvicorn
import asyncio
import cv2
from vidgear.gears.asyncio import WebGear
from vidgear.gears.asyncio.helper import reducer
import cv2 as cv
import mediapipe as mp
import time
import utils
import math
import numpy as np


frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0

CLOSED_EYES_FRAME = 3
FONTS = cv.FONT_HERSHEY_COMPLEX

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

map_face_mesh = mp.solutions.face_mesh

web = WebGear(logging=True)

def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height))
                  for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

    # retorna a lista de tuplas para cada landmarks
    return mesh_coord

def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def blinkRatio(img, landmarks, right_indices, left_indices):
    # OLHOS DIREITOS
    # linha horizontal
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # linha vertical
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # desenha linhas nos olhos direitos
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # OLHOS ESQUERDOS
    # linha horizontal
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # linha vertical
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio

def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # converte imagem colorida para escala de cinza
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # obtem a dimensão da imagem
    dim = gray.shape

    # cria máscara de escala de cinza dim
    mask = np.zeros(dim, dtype=np.uint8)

    # desenha a forma dos olhos na máscara
    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # showing the mask
    # cv.imshow('mask', mask)

    # draw eyes image on mask, where white shape is
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    # change black color to gray other than eys
    # cv.imshow('eyes draw', eyes)
    eyes[mask == 0] = 155

    # getting minium and maximum x and y  for right and left eyes
    # For Right Eye
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # For LEFT Eye
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # croping the eyes from mask
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # returning the cropped eyes
    return cropped_right, cropped_left

def positionEstimator(cropped_eye):
    # getting height and width of eye
    h, w = cropped_eye.shape

    # remove the noise from images
    gaussain_blur = cv.GaussianBlur(cropped_eye, (9, 9), 0)
    median_blur = cv.medianBlur(gaussain_blur, 3)

    # applying thrsholding to convert binary_image
    ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)

    # create fixd part for eye with
    piece = int(w/3)

    # slicing the eyes into three parts
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece+piece]
    left_piece = threshed_eye[0:h, piece + piece:w]

    # calling pixel counter function
    eye_position, color = pixelCounter(right_piece, center_piece, left_piece)

    return eye_position, color

def pixelCounter(first_piece, second_piece, third_piece):
    # counting black pixel in each part
    right_part = np.sum(first_piece == 0)
    center_part = np.sum(second_piece == 0)
    left_part = np.sum(third_piece == 0)
    # creating list of these values
    eye_parts = [right_part, center_part, left_part]

    # getting the index of max values in the list
    max_index = eye_parts.index(max(eye_parts))
    pos_eye = ''
    if max_index == 0:
        pos_eye = "RIGHT"
        color = [utils.BLACK, utils.GREEN]
    elif max_index == 1:
        pos_eye = 'CENTER'
        color = [utils.YELLOW, utils.PINK]
    elif max_index == 2:
        pos_eye = 'LEFT'
        color = [utils.GRAY, utils.YELLOW]
    else:
        pos_eye = "Closed"
        color = [utils.GRAY, utils.YELLOW]
    return pos_eye, color


with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    # starting time here 
    start_time = time.time()

    # create your own custom frame producer
    async def my_frame_producer():

        # !!! define your own video source here !!!
        # Open any video stream such as live webcam
        # video stream on first index(i.e. 0) device
        stream = cv2.VideoCapture(0)
        # loop over frames
        while True:
            # read frame from provided source
            (grabbed, frame) = stream.read()
            # break if NoneType
            if not grabbed:
                break

            # do something with your OpenCV frame here
            
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            results  = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_coords = landmarksDetection(frame, results, False)
                #frame =utils.fillPolyTrans(frame, [mesh_coords[p] for p in FACE_OVAL], utils.WHITE, opacity=0.2)
                frame =utils.fillPolyTrans(frame, [mesh_coords[p] for p in LEFT_EYE], utils.GREEN, opacity=0.1)
                frame =utils.fillPolyTrans(frame, [mesh_coords[p] for p in RIGHT_EYE], utils.GREEN, opacity=0.1)
                #frame =utils.fillPolyTrans(frame, [mesh_coords[p] for p in LEFT_EYEBROW], utils.WHITE, opacity=0.1)
                #frame =utils.fillPolyTrans(frame, [mesh_coords[p] for p in RIGHT_EYEBROW], utils.WHITE, opacity=0.1)
                #frame =utils.fillPolyTrans(frame, [mesh_coords[p] for p in LIPS], utils.WHITE, opacity=0.1 )
    
            # calculating  frame per seconds FPS
            end_time = time.time()-start_time
            fps = frame_counter/end_time

            #frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (20, 50), bgOpacity=0.9, textThickness=2)

            # reducer frames size if you want more performance otherwise comment this line
            # reduce frame by 30%
            frame = await reducer(frame, percentage=30, interpolation=cv2.INTER_AREA)
            # handle JPEG encoding
            encodedImage = cv2.imencode(".jpg", frame)[1].tobytes()
            # yield frame in byte format
            yield (b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + encodedImage + b"\r\n")
            await asyncio.sleep(0)
        # close stream
        stream.release()


    # add your custom frame producer to config
    web.config["generator"] = my_frame_producer

    # run this app on Uvicorn server at address http://localhost:8000/
    uvicorn.run(web(), host="localhost", port=8000)

    # close app safely
    web.shutdown()
