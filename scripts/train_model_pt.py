#!/usr/bin/env python
"""
Train a lightweight MLP on hand‑keypoint vectors with PyTorch,
export to ONNX, and save label map + normaliser + binding map.
Run:  python scripts/train_model_pt.py
"""

import pathlib, pickle, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = pathlib.Path("data/processed")
MODEL_DIR = pathlib.Path("models"); MODEL_DIR.mkdir(exist_ok=True)

# ---------------------  dataset utils  -----------------------------
def load_data():
    csv_files = list(DATA_DIR.glob("*.csv"))
    print(f"📂 Found {len(csv_files)} CSVs in {DATA_DIR}")
    if not csv_files:
        raise ValueError("❌ No CSV gesture data found in data/processed/")

    frames = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)

    if df.empty:
        raise ValueError("❌ CSVs are empty — cannot train on empty dataset.")

    X = df.iloc[:, 2:].values.astype(np.float32)
    y = df.label.values

    classes = np.unique(y)
    lbl2id = {c: i for i, c in enumerate(classes)}
    y_num = np.vectorize(lbl2id.get)(y)

    # ✅ Add binding map
    binding_map = {row["label"]: row["binding"] for _, row in df.iterrows()}

    return X, y_num, lbl2id, binding_map

X, y, lbl2id, binding_map = load_data()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler().fit(Xtr)
Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

Xtr = torch.tensor(Xtr, dtype=torch.float32)
ytr = torch.tensor(ytr, dtype=torch.long)
Xte = torch.tensor(Xte, dtype=torch.float32)
yte = torch.tensor(yte, dtype=torch.long)


# ------------------------  model  ----------------------------------
class KeypointMLP(nn.Module):
    def __init__(self, in_dim=63, n_classes=len(lbl2id)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)

model = KeypointMLP()
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 60
batch = 64

# ------------------------  training  -------------------------------
for epoch in range(epochs):
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), batch):
        idx = perm[i:i+batch]
        out = model(Xtr[idx])
        loss = criterion(out, ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            acc = (model(Xte).argmax(1) == yte).float().mean().item() * 100
        print(f"epoch {epoch+1:02d}/{epochs} – val acc {acc:.2f}%")

# ------------------------  export  ---------------------------------
pt_path   = MODEL_DIR / "gesture_clf_pt.pt"
onnx_path = MODEL_DIR / "gesture_clf_pt.onnx"
meta_path = MODEL_DIR / "meta_pt.pkl"

# Save PyTorch model
torch.save(model.state_dict(), pt_path)

# Save ONNX model
dummy = torch.randn(1, 63)
torch.onnx.export(model, dummy, onnx_path,
                  input_names=["float_input"], output_names=["output"],
                  dynamic_axes={"float_input": {0: "batch"}})

# Save scaler, label_map, and binding_map
with open(meta_path, "wb") as f:
    pickle.dump({
        "scaler": scaler,
        "label_map": lbl2id,
        "binding_map": binding_map
    }, f)

print("✅ Saved:")
print(f"→ Model (PyTorch):       {pt_path}")
print(f"→ Model (ONNX):          {onnx_path}")
print(f"→ Metadata (Pickle):     {meta_path}")
