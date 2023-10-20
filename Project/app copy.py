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
# import sys



app = Flask(__name__)

### Dlib 얼굴 검출기 초기화 ************************************
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return faces, gray

def draw_faces(frame, faces, gray):
    # 여러 얼굴에 대한 비율을 저장할 리스트
    ratios = []
    
    for face in faces:
        #x,y값은 밑에서 다시 쓰니까 빼도됨.
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()

        # 해당 영역은 감지된 얼굴을 사각형으로 표현해주는 영역
        # 감지된 얼굴의 영역에 landmarks를 표현 할거니까 사각형은 일단 감춰둔다. 
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        # 'part' 메서드를 사용하여 랜드마크 포인트에 접근하고, 각 점의 'x'와 'y' 좌표를 얻음
        left_point = (landmarks.part(36).x, landmarks.part(36).y)  # .part 메서드 사용
        right_point = (landmarks.part(39).x, landmarks.part(39).y)  # .part 메서드 사용
        center_top =  midpoint(landmarks.part(37), landmarks.part(38)) # .part 메서드 사용
        center_bottom = midpoint(landmarks.part(41), landmarks.part(40)) # .part 메서드 사용
        
        
        # 비율 계산
        hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        if ver_line_length > 0:  # 0으로 나누는 것을 방지
            ratio = hor_line_length / ver_line_length
            ratios.append(ratio)

    # 이제 여러 얼굴의 비율을 처리할 수 있습니다.
    return ratios

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
    # Receive image data from the client
    blob = request.data
    nparr = np.frombuffer(blob, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform object detection (for example, face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    try :
        faces, gray = detect_faces(frame)
        ratios = draw_faces(frame, faces, gray)

        # 각 얼굴의 비율을 출력하거나 다른 처리를 할 수 있습니다.
        for ratio in ratios:
            print("Ratio: {:.2f}".format(ratio))
    except :
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        response = Response(frame, content_type='image/jpg')
        return response

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