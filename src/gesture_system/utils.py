import numpy as np
import os
import pathlib

def extract_hand_keypoints(results):
    """
    Given a Mediapipe Holistic result, return a 63-element (21×(x,y,z)) numpy array
    or None if no hand is detected.
    """
    hand_landmarks = None
    if results.left_hand_landmarks:
        hand_landmarks = results.left_hand_landmarks.landmark
    elif results.right_hand_landmarks:
        hand_landmarks = results.right_hand_landmarks.landmark

    if not hand_landmarks:
        return None

    return np.array([[p.x, p.y, p.z] for p in hand_landmarks], dtype=np.float32).flatten()

def save_keypoints(vec, gesture_name, kb_key, out_dir):
    """
    Append a single keypoint vector to CSV in out_dir/gesture_name.csv,
    creating header if needed.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = pathlib.Path(out_dir) / f"{gesture_name}.csv"
    cols = ["label", "binding"] + [f"kp{i}" for i in range(len(vec))]
    header = ",".join(cols)
    line = ",".join([gesture_name, kb_key] + [f"{v:.6f}" for v in vec])

    new_file = not path.exists()
    with path.open("a") as f:
        if new_file:
            f.write(header + "\n")
        f.write(line + "\n")
