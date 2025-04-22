import cv2
import numpy as np
import mediapipe as mp
import time

class FaceService:
    def __init__(self, detection_confidence=0.6, reauth_interval=5.0):
        self.cap = cv2.VideoCapture(0)
        self.reauth_interval = reauth_interval
        self.last_check = 0
        self.authenticated = False

        # Mediapipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=detection_confidence
        )

        # Save a dummy "embedding" from registered face
        self.reference_landmarks = None

    def read_frame(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def register_face(self, frame):
        """
        Capture and store the face landmarks of the user.
        """
        results = self.face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            # Use bounding box center as simple "embedding"
            detection = results.detections[0]
            box = detection.location_data.relative_bounding_box
            self.reference_landmarks = (box.xmin + box.width / 2, box.ymin + box.height / 2)
            print("✅ Face registered.")
            return True
        print("❌ No face detected during registration.")
        return False

    def is_authenticated(self, frame):
        now = time.time()
        if now - self.last_check > self.reauth_interval:
            self.authenticated = self._check_face(frame)
            self.last_check = now
        return self.authenticated

    def _check_face(self, frame):
        if self.reference_landmarks is None:
            return False

        results = self.face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            detection = results.detections[0]
            box = detection.location_data.relative_bounding_box
            current = (box.xmin + box.width / 2, box.ymin + box.height / 2)
            # Euclidean distance from registered center
            dist = np.linalg.norm(np.array(current) - np.array(self.reference_landmarks))
            return dist < 0.1  # Threshold for matching
        return False

    def force_reauth(self):
        self.last_check = 0

    def __del__(self):
        self.cap.release()
