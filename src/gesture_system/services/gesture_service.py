import time
import cv2
import joblib
import onnxruntime
from gesture_system.utils import extract_hand_keypoints, mp_holistic
import keyboard

class GestureService:
    def __init__(self, model_path="models/gesture_clf_pt.onnx", meta_path="models/meta_pt.pkl"):
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

        self.holo = mp_holistic.Holistic(static_image_mode=False, model_complexity=0)

        self.pause = False
        self.last_time = 0.0
        self.delay = 2.0
        self.ctrl_cooldown = 1.0
        self.ctrl_until = 0.0

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.holo.process(rgb)
        kp = extract_hand_keypoints(res)
        if kp is None:
            return None, None

        now = time.time()
        X = self.scaler.transform(kp.reshape(1, -1).astype("float32"))
        pred = self.sess.run(None, {"float_input": X})[0].argmax()
        gesture = self.rev_label[int(pred)]
        binding = self.binding_map.get(gesture, "").lower()

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

    def toggle_pause(self):
        self.pause = not self.pause

    def resume(self):
        self.pause = False