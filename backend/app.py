# backend/app.py
import sys
import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# allow imports from src/gesture_system
ROOT = os.path.dirname(__file__)
SRC  = os.path.abspath(os.path.join(ROOT, "..", "src"))
sys.path.append(SRC)

from gesture_system.services.face_service    import FaceService
from gesture_system.services.gesture_service import GestureService

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

face_svc = FaceService()
gest_svc = GestureService()

@app.post("/register_face")
async def register_face(
    file: UploadFile = File(...),
    username: str   = Form(...),
):
    data = await file.read()
    arr  = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Invalid image")

    # delegate to your service
    try:
        result = face_svc.register_user(username, frame)
        return {"status": "ok", **result}
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/collect_gesture")
async def collect_gesture(
    file: UploadFile    = File(...),
    gesture_name: str   = Form(...),
    binding: str        = Form(...),
):
    data = await file.read()
    arr  = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Invalid image")

    try:
        gest_svc.save_keypoint_from_frame(frame, gesture_name, binding)
        return {"status": "collected", "gesture": gesture_name}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/train_model")
def train_model():
    # dynamic import so we can find scripts/ even from inside backend/
    sys.path.append(os.path.abspath(os.path.join(ROOT, "..")))
    from scripts.train_model_pt import train_model_pt
    train_model_pt()
    return {"status": "training complete"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    arr  = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Invalid image")

    try:
        gesture, binding = gest_svc.process_frame(frame)
        return {"gesture": gesture, "binding": binding}
    except FileNotFoundError:
        raise HTTPException(503, "Model not found; please train first")
