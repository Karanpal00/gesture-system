import cv2
import numpy as np
import time
import onnxruntime
import joblib
import keyboard

# 🌟 FIXED IMPORT PATH — was "from .utils" but utils lives up one level
from gesture_system.utils import extract_hand_keypoints

class GestureService:
    def __init__(
        self,
        model_path="models/gesture_clf_pt.onnx",
        meta_path="models/meta_pt.pkl"
    ):
        providers = []
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.sess = onnxruntime.InferenceSession(model_path, providers=providers)

        meta = joblib.load(meta_path)
        self.scaler = meta["scaler"]
        self.label_map = meta["label_map"]
        self.binding_map = meta["binding_map"]
        self.rev_label = {v: k for k, v in self.label_map.items()}

        self.pause = False
        self.last_time = 0.0
        self.delay = 2.0
        self.ctrl_cooldown = 1.0
        self.ctrl_until = 0.0

    def process_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # reuse Mediapipe holistic from utils if you like…

        # extract keypoints
        kp = extract_hand_keypoints(self._holistic.process(rgb))
        if kp is None:
            return None, None

        now = time.time()
        X = self.scaler.transform(kp.reshape(1, -1).astype("float32"))
        pred = self.sess.run(None, {"float_input": X})[0].argmax()
        gesture = self.rev_label[int(pred)]
        binding = self.binding_map.get(gesture, "").lower()

        # handle control gestures
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

        if self.pause or not binding or binding == "none":
            return gesture, None

        if now - self.last_time >= self.delay:
            keyboard.press_and_release(binding)
            self.last_time = now
            return gesture, binding

        return gesture, None
