from fastapi import FastAPI, WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from starlette.requests import Request
import asyncio
import cv2

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
                
async def process_video_analysis(websocket):
    camera_enabled = True  # 카메라 활성화 상태를 표시하는 변수

    while camera_enabled:
        result = "Some analysis result"  # 예시 결과
        await websocket.send_text(result)

        # 클라이언트로부터 "stop_camera" 메시지가 오면 카메라를 끕니다.
        message = await websocket.receive_text()
        if message == "stop_camera":
            camera_enabled = False
            break

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await process_video_analysis(websocket)

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("cam.html", {"request": request})

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_bytes()
        key = cv2.waitKey(5)
        if key == 27:
            break
    return templates.TemplateResponse("cam.html",{data:data})