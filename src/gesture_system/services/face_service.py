import cv2
from pathlib import Path
from mediapipe import solutions as mp_solutions

class FaceService:
    """
    A very lightweight “authentication” that just checks
    for *any* face in the frame via MediaPipe.
    The `register_user` endpoint simply saves a reference image.
    """
    def __init__(self, data_dir: str = "data/faces", min_confidence: float = 0.5):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.detector = mp_solutions.face_detection.FaceDetection(
            min_detection_confidence=min_confidence
        )

    def register_user(self, username: str, frame):
        """
        Save the uploaded frame under data/faces/<username>.png
        """
        out_path = self.data_dir / f"{username}.png"
        success = cv2.imwrite(str(out_path), frame)
        if not success:
            raise ValueError("Could not write face image to disk.")
        return {"user": username, "path": str(out_path)}

    def is_authenticated(self, frame) -> bool:
        """
        True if MediaPipe sees *any* face in the frame.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.process(rgb)
        return bool(result.detections)
