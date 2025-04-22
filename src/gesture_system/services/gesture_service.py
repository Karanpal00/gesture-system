# src/gesture_system/services/gesture_service.py
import pathlib
import time
import joblib
import onnxruntime
from .utils import extract_hand_keypoints
import keyboard

class GestureService:
    def __init__(
        self,
        model_file: str = "gesture_clf_pt.onnx",
        meta_file:  str = "meta_pt.pkl",
    ):
        # locate backend/models/
        project_root = pathlib.Path(__file__).resolve().parent.parent.parent
        model_dir    = project_root / "backend" / "models"
        model_path   = model_dir / model_file
        meta_path    = model_dir / meta_file

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        # ONNX runtime
        providers = []
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.sess = onnxruntime.InferenceSession(str(model_path), providers=providers)

        meta = joblib.load(str(meta_path))
        self.scaler     = meta["scaler"]
        self.label_map  = meta["label_map"]
        self.binding_map= meta["binding_map"]
        self.rev_label  = {v:k for k,v in self.label_map.items()}

        # timing & state
        self.delay     = 2.0
        self.last_time = 0.0

    def process_frame(self, frame):
        """Return (gesture_label, binding_key) or (None, None)."""
        kp = extract_hand_keypoints(frame)
        if kp is None:
            return None, None

        X = self.scaler.transform(kp.reshape(1,-1))
        pred = self.sess.run(None, {"float_input": X.astype("float32")})[0]
        idx  = int(pred.argmax())
        gesture = self.rev_label[idx]
        binding = self.binding_map.get(gesture, None)

        now = time.time()
        if binding and binding != "none" and (now - self.last_time) >= self.delay:
            keyboard.press_and_release(binding)
            self.last_time = now

        return gesture, binding

    def save_keypoint_from_frame(self, frame, gesture_name, binding):
        """Append one keypoint vector to data/raw/*.csv"""
        from ..utils import save_keypoints
        kp = extract_hand_keypoints(frame)
        if kp is None:
            raise ValueError("No hand detected")
        out_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "raw"
        save_keypoints(kp, gesture_name, binding, str(out_dir))
