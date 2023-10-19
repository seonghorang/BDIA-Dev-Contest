from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from starlette.requests import Request
import cv2
import numpy as np

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# OpenCV로 웹캠 스트림을 가져와서 이미지로 변환
def capture_webcam():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        ret, image = cv2.imencode(".png", frame)
        if ret:
            return image.tobytes()
    return None

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 웹캠 이미지를 업데이트하는 엔드포인트 추가
@app.get("/update_image")
async def update_image():
    image = capture_webcam()
    return HTMLResponse(content=image, status_code=200)