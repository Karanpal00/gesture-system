import cv2
import face_recognition
import numpy as np
import pathlib

class FaceService:
    def __init__(self, tolerance=0.45, reauth_interval=5.0):
        self.cap = cv2.VideoCapture(0)
        self.tolerance = tolerance
        self.reauth_interval = reauth_interval
        self.last_check = 0
        self.authenticated = False

        # Load known face embeddings
        self.known_embs = []
        face_dir = pathlib.Path("data/faces")
        face_dir.mkdir(parents=True, exist_ok=True)
        for f in face_dir.glob("*.npy"):
            self.known_embs.append(np.load(f))

    def read_frame(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def is_authenticated(self, frame):
        now = cv2.getTickCount() / cv2.getTickFrequency()
        if now - self.last_check > self.reauth_interval:
            self.authenticated = self._check_face(frame)
            self.last_check = now
        return self.authenticated

    def _check_face(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)
        for enc in encs:
            matches = face_recognition.compare_faces(self.known_embs, enc, self.tolerance)
            if any(matches):
                return True
        return False

    def force_reauth(self):
        self.last_check = 0

    def __del__(self):
        self.cap.release()