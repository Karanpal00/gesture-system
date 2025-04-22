# src/gesture_system/utils.py
import os
import pathlib
import numpy as np

def extract_hand_keypoints(frame: np.ndarray):
    """Use MediaPipe Holistic to get 63‐element hand vector or None."""
    import cv2
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
    ) as holo:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holo.process(rgb)
        lm = None
        if res.left_hand_landmarks:
            lm = res.left_hand_landmarks.landmark
        elif res.right_hand_landmarks:
            lm = res.right_hand_landmarks.landmark
        if not lm:
            return None
        return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()

def save_keypoints(vec, gesture_name, kb_key, out_dir):
    """
    Append a single keypoint vector to CSV in out_dir/<gesture_name>.csv,
    creating header if needed.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = pathlib.Path(out_dir) / f"{gesture_name}.csv"
    cols = ["label", "binding"] + [f"kp{i}" for i in range(len(vec))]
    header = ",".join(cols)
    line   = ",".join([gesture_name, kb_key] + [f"{v:.6f}" for v in vec])

    new_file = not path.exists()
    with open(path, "a") as f:
        if new_file:
            f.write(header + "\n")
        f.write(line + "\n")
