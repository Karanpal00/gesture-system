from fastapi import FastAPI, WebSocket
from gesture_system.services.face_service import FaceService
from gesture_system.services.gesture_service import GestureService
import cv2
import base64
import numpy as np

app = FastAPI()
face = FaceService()
gest = GestureService()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        frame_data = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
        frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

        if not face.is_authenticated(frame):
            await websocket.send_text("LOCKED")
            continue

        gesture, binding = gest.process(frame)
        await websocket.send_json({
            "gesture": gesture,
            "binding": binding
        })
