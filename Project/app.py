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

# 눈 깜박임 비율을 저장할 리스트
blink_ratios_history = []


# 깜박임 감지를 위한 임계값 설정
BLINK_RATIO_THRESHOLD = 4.8 # 이 값은 실험을 통해 적절한 값을 찾아야 합니다.
CLOSED_EYES_FRAME_THRESHOLD = 9  # 눈을 감은 것으로 간주할 프레임 수

# 눈을 감은 프레임을 추적하기 위한 변수
closed_eyes_frame_counter = 0


# 졸림 상태 및 프레임 카운터를 추적하기 위한 변수
drowsy_alert_active = False
drowsy_frame_counter = 0

# 랜드마크의 중간점을 계산하는 함수
def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

# 시선 비율을 계산하는 함수
def get_gaze_ratio(eye_points, facial_landmarks, frame, gray):
    # 눈 영역의 좌표를 구합니다.
    eye_region = np.array([(facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in eye_points])

    # 눈 영역에서 최소/최대 x 및 y 좌표를 찾아 눈 영역을 만듭니다.
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    # 그레이스케일 이미지에서 눈 영역을 추출합니다.
    eye = gray[min_y: max_y, min_x: max_x]

    # 눈 영역의 너비와 높이가 충분히 큰지 확인합니다.
    if eye.size == 0 or eye.shape[0] < 2 or eye.shape[1] < 2:
        return None

    # 눈동자 감지를 위해 threshold를 적용합니다.
    _, eye = cv2.threshold(eye, 70, 255, cv2.THRESH_BINARY_INV)
    
    # 눈동자의 위치를 찾기 위한 코드 (예: 중심 찾기, 가중치 적용 등)
    # 이 부분은 실제 눈동자 위치를 정확하게 찾는 로직으로 변경되어야 합니다.
    # 예시로, 단순하게 흰색 픽셀(눈동자)의 수를 계산하는 것을 사용할 수 있습니다.
    white_pixels = cv2.countNonZero(eye)

    # 눈 영역의 총 픽셀 수를 계산합니다.
    total_pixels = eye.shape[0] * eye.shape[1]

    # 감지된 흰색 픽셀의 비율을 계산합니다.
    gaze_ratio = white_pixels / total_pixels if total_pixels > 0 else 0

    return gaze_ratio
    
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return faces, gray

# 주요 기능을 수행하는 함수: 얼굴 감지, 랜드마크 추출, 비율 계산
def draw_faces(frame, faces, gray):
    blink_ratios = []
    gaze_ratios = []
    nose_ratios = []

    for face in faces:
        landmarks = predictor(gray, face)

        # 깜박임 감지 부분의 코드
        left_point = (landmarks.part(36).x, landmarks.part(36).y)  # .part 메서드 사용
        right_point = (landmarks.part(39).x, landmarks.part(39).y)  # .part 메서드 사용
        center_top =  midpoint(landmarks.part(37), landmarks.part(38)) # .part 메서드 사용
        center_bottom = midpoint(landmarks.part(41), landmarks.part(40)) # .part 메서드 사용
        
        
        # 비율 계산
        hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        if ver_line_length > 0:  # 0으로 나누는 것을 방지
            ratio = hor_line_length / ver_line_length
            blink_ratios.append(ratio)

        # 시선 감지 부분의 코드
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, frame, gray)  # 매개변수 추가
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, frame, gray)  # 매개변수 추가
        gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye) / 2
        gaze_ratios.append(gaze_ratio)

        ### 머리 회전 반경 계산
        end_nose_point = (landmarks.part(29).x, landmarks.part(29).y)
        left_libs_point = (landmarks.part(4).x, landmarks.part(4).y)
        right_libs_point = (landmarks.part(12).x, landmarks.part(12).y)

        #코와 입의 길이 계산
        nose_line_len_left = hypot(left_libs_point[0]-end_nose_point[0],left_libs_point[1]-end_nose_point[1])
        nose_line_len_right = hypot(right_libs_point[0]-end_nose_point[0],right_libs_point[1]-end_nose_point[1])
        nose_ratio = nose_line_len_left/nose_line_len_right 
        nose_ratios.append(nose_ratio)

    return blink_ratios, gaze_ratios, nose_ratios

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
    global closed_eyes_frame_counter  # 전역 변수를 함수 내에서 사용할 수 있도록 선언
     
    # Receive image data from the client
    blob = request.data
    nparr = np.frombuffer(blob, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform object detection (for example, face detection)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # try :
    faces, gray = detect_faces(frame)
    blink_ratios, gaze_ratios, nose_ratios = draw_faces(frame, faces, gray)
    
    for blink_ratio in blink_ratios:
        if blink_ratio > BLINK_RATIO_THRESHOLD:
            # 눈을 감은 것으로 간주하고 카운터를 증가시킵니다.
            closed_eyes_frame_counter += 1
        else:
            # 눈이 감기지 않았으면 카운터를 초기화합니다.
            closed_eyes_frame_counter = 0

        # 일정 시간 동안 눈을 감은 경우 졸음 상태로 간주합니다.
        if closed_eyes_frame_counter >= CLOSED_EYES_FRAME_THRESHOLD:
            print("졸림 상태 경고!")
            # 여기에 경고 상태를 유지하고 싶다면 추가적인 로직을 구현해야 합니다.

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