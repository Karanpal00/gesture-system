import cv2
import numpy as np
import pathlib
import onnxruntime
import joblib
import time
from gesture_system.utils import mp_holistic  # use mediapipe-only face detection here

class FaceService:
    def __init__(self, tolerance=0.45, reauth_interval=5.0):
        self.cap = cv2.VideoCapture(0)
        self.tolerance = tolerance
        self.reauth_interval = reauth_interval
        self.last_check = 0
        self.authenticated = False

        # set up mediapipe face detection
        self.face_detector = mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=0,
        )

        # load saved embeddings
        self.face_dir = pathlib.Path("data/faces")
        self.face_dir.mkdir(exist_ok=True, parents=True)
        self.known_embs = []
        for f in self.face_dir.glob("*.npy"):
            self.known_embs.append(np.load(f))

    def read_frame(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def is_authenticated(self, frame):
        now = time.time()
        if now - self.last_check > self.reauth_interval:
            self.authenticated = self._check_face(frame)
            self.last_check = now
        return self.authenticated

    def _check_face(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_detector.process(rgb)
        # Here you'd extract landmarks → embedding; for simplicity assume success if any face detected:
        if res.face_landmarks:
            # (in real use, compute embedding and compare to self.known_embs)
            return True
        return False

    def register_user(self, username: str, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_detector.process(rgb)
        if not res.face_landmarks:
            raise ValueError("No face detected")
        # dummy embedding: flatten landmarks
        emb = np.array([[lm.x, lm.y, lm.z] for lm in res.face_landmarks.landmark], dtype=np.float32).flatten()
        path = self.face_dir / f"{username}.npy"
        np.save(path, emb)
        return {"user": username}

    def __del__(self):
        self.cap.release()
