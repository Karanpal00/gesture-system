import time
import keyboard
import numpy as np
import joblib
import onnxruntime
from .utils import extract_hand_keypoints
from mediapipe import solutions as mp_solutions

class GestureService:
    """
    Wraps your ONNX inference + mediapipe hand‐keypoint extraction.
    """
    def __init__(
        self,
        model_path: str = "models/gesture_clf_pt.onnx",
        meta_path: str = "models/meta_pt.pkl"
    ):
        # ONNX Runtime with GPU→CPU fallback
        providers = []
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.sess = onnxruntime.InferenceSession(model_path, providers=providers)

        # Load metadata
        meta = joblib.load(meta_path)
        self.scaler      = meta["scaler"]
        self.label_map   = meta["label_map"]
        self.binding_map = meta["binding_map"]
        self.rev_label   = {v: k for k, v in self.label_map.items()}

        # Mediapipe holistic for hand keypoints
        self.holo = mp_solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=0
        )

        # Throttling state
        self.pause     = False
        self.last_time = 0.0
        self.delay     = 2.0
        self.ctrl_until    = 0.0
        self.ctrl_cooldown = 1.0

    def process_frame(self, frame):
        """
        Run one pass: extract keypoints, infer, apply binding (if not paused).
        Returns (gesture_name, binding_key or None).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.holo.process(rgb)
        kp  = extract_hand_keypoints(res)
        if kp is None:
            return None, None

        now = time.time()
        # model input
        X    = self.scaler.transform(kp.reshape(1, -1).astype("float32"))
        pred = self.sess.run(None, {"float_input": X})[0].argmax()
        gesture = self.rev_label[int(pred)]
        binding = self.binding_map.get(gesture, "").lower()

        # control gestures
        if now >= self.ctrl_until:
            if gesture == "pause_input":
                self.pause = True
                self.ctrl_until = now + self.ctrl_cooldown
            elif gesture == "resume_input":
                self.pause = False
                self.ctrl_until = now + self.ctrl_cooldown
            elif gesture == "reduce_delay":
                self.delay = 1.0
                self.ctrl_until = now + self.ctrl_cooldown

        # if paused or no binding, skip keypress
        if self.pause or not binding or binding == "none":
            return gesture, None

        # throttle normal gestures
        if now - self.last_time >= self.delay:
            keyboard.press_and_release(binding)
            self.last_time = now
            return gesture, binding

        return gesture, None
