# src/gesture_system/services/face_service.py
import pathlib
import numpy as np
import mediapipe as mp

class FaceService:
    def __init__(self, tolerance: float = 0.45, reauth_interval: float = 5.0):
        # locate backend/data/faces
        project_root = pathlib.Path(__file__).resolve().parent.parent.parent
        faces_dir    = project_root / "backend" / "data" / "faces"
        faces_dir.mkdir(parents=True, exist_ok=True)
        self.faces_dir = faces_dir

        # load any saved embeddings
        self.known_embs = [
            np.load(f) for f in faces_dir.glob("*.npy")
        ]

        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.tolerance       = tolerance
        self.reauth_interval = reauth_interval
        self.last_check      = 0
        self.authenticated   = False

    def register_user(self, username: str, frame: np.ndarray):
        """Detect first face, take its embedding, save as <username>.npy"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        if not results.detections:
            raise ValueError("No face detected")
        # pick first detection box
        box = results.detections[0].location_data.relative_bounding_box
        h, w, _ = frame.shape
        x1 = int(box.xmin * w)
        y1 = int(box.ymin * h)
        x2 = x1 + int(box.width * w)
        y2 = y1 + int(box.height * h)
        face_roi = frame[y1:y2, x1:x2]
        # convert ROI to embedding via mediapipe face mesh → approximate
        # here you'd insert your embedding routine or fallback to face_recognition
        # For simplicity, we'll just flatten the ROI as a placeholder:
        emb = cv2.resize(face_roi, (64,64)).flatten().astype(np.float32)
        out_path = self.faces_dir / f"{username}.npy"
        np.save(out_path, emb)
        self.known_embs.append(emb)
        return {"user": username}

    # (optionally add methods to re-auth during predict if you want)
