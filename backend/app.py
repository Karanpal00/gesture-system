from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from gesture_system.services.face_service import FaceService
from gesture_system.services.gesture_service import GestureService
from scripts.train_model_pt import train_model_pt  # import your training script
import cv2
import numpy as np

app = FastAPI()

# CORS setup for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate face and gesture services
face = FaceService()
gesture = GestureService()

@app.post("/register_face")
async def register_face(file: UploadFile, username: str = Form(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    face.register_user(username, frame)
    return {"status": "success", "user": username}

@app.post("/collect_gesture")
async def collect_gesture(file: UploadFile, gesture_name: str = Form(...), binding: str = Form(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    gesture.save_keypoint_from_frame(frame, gesture_name, binding)
    return {"status": "collected", "gesture": gesture_name}

@app.post("/train_model")
def train_model():
    train_model_pt()  # Train and export updated model
    return {"status": "training complete"}

@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    label, binding = gesture.process(frame)
    return {"gesture": label, "binding": binding}
