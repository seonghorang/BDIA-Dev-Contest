from flask import Flask, render_template, Response, jsonify, request
import numpy as np

import cv2
import dlib
import numpy as np
import time
import datetime
from datetime import datetime, timedelta
import math
from math import hypot
from scipy.spatial import distance
from collections import Counter
from scipy.spatial import distance as dist
# import sys



app = Flask(__name__)

### Dlib 얼굴 검출기 초기화 ************************************
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

EYE_AR_CONSEC_FRAMES = 90  # 눈을 감고 있어야 하는 연속 프레임 수

COUNTER = 0  # 눈을 감고 있는 프레임 수를 세는 데 사용되는 카운터

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_eye_aspect_ratio(eye_points, facial_landmarks):
    # Calculate the euclidean distances between the two sets of 
    # vertical eye landmarks (x, y)-coordinates
    corner_left = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    corner_right = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)

    top_center = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    bottom_center = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_length = dist.euclidean(corner_left, corner_right)
    ver_line_length = dist.euclidean(top_center, bottom_center)

    # compute the eye aspect ratio
    if ver_line_length != 0:
        ear = hor_line_length / ver_line_length
    else:
        ear = hor_line_length

    return ear

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return faces, gray

@app.route('/')
def index():
    return render_template('/video/main.html')

### 영상 구동 페이진
@app.route('/video_feed')
def video_feed():
    return render_template('/video/video.html')


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/process', methods=['POST'])
def process():
    global COUNTER
     
    # Receive image data from the client
    blob = request.data
    nparr = np.frombuffer(blob, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform object detection (for example, face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # try :
    faces, gray = detect_faces(frame)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye_ratio = get_eye_aspect_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_eye_aspect_ratio([42, 43, 44, 45, 46, 47], landmarks)

        # We will consider the average of both eyes
        blink_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blink_ratio > 4.5:  # The eye is closed if the ratio is more than 3.2
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                print("DROWSINESS ALERT!")
                COUNTER = 0  # 눈이 감긴 프레임 수를 초기화

        else:
            COUNTER = 0
    # except :
    #     _, buffer = cv2.imencode('.jpg', frame)
    #     frame = buffer.tobytes()
        
    #     response = Response(frame, content_type='image/jpg')
    #     return response

    #################################################

    # 프레임을 클라이언트로 전송
    # ret, jpeg = cv2.imencode('.jpg', frame)        
    _, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    
    response = Response(frame, content_type='image/jpg')
    return response


if __name__ == '__main__':
    # app.run(debug=True)
    
    # debug=True는 에러가 없으면, 자동으로 서버 재시작함
    # 코드 수정 시 애러가 없으면, 서버가 재 실행 됨
    app.debug = True 
    
    # run에는 여러가지 옵션이 있음
    # app.run(host="0.0.0.0", port="5000", debug=True)
    app.run(host="127.0.0.1", port="5000")