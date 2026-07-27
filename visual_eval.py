import cv2
import mediapipe as mp
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import models

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

base_dir = r"C:\Users\91636\project 1\Celeb-DF-v2"

def extract_face_crops(video_path, max_frames=3):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    crops = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)
            face_crop = frame[y_min:y_max, x_min:x_max]
            if face_crop.size > 0:
                face_crop = cv2.resize(face_crop, (224, 224))
                crops.append(face_crop)
    cap.release()
    return crops

# ---- Use videos 101-130 (NOT used in training, which used 0-100) ----
real_dir = os.path.join(base_dir, "Celeb-real")
fake_dir = os.path.join(base_dir, "Celeb-synthesis")

real_files = os.listdir(real_dir)[100:130]
fake_files = os.listdir(fake_dir)[100:130]

X_eval = []
y_eval = []

print("Processing UNSEEN real videos...")
for i, fname in enumerate(real_files):
    path = os.path.join(real_dir, fname)
    crops = extract_face_crops(path, max_frames=3)
    for crop in crops:
        X_eval.append(crop)
        y_eval.append(0)
    print(f"  {i+1}/{len(real_files)}: {fname} -> {len(crops)} crops")

print("Processing UNSEEN fake videos...")
for i, fname in enumerate(fake_files):
    path = os.path.join(fake_dir, fname)
    crops = extract_face_crops(path, max_frames=3)
    for crop in crops:
        X_eval.append(crop)
        y_eval.append(1)
    print(f"  {i+1}/{len(fake_files)}: {fname} -> {len(crops)} crops")

X_eval = np.array(X_eval).astype(np.float32) / 255.0
X_eval = np.transpose(X_eval, (0, 3, 1, 2))
y_eval = np.array(y_eval)

print("\nEval set shape:", X_eval.shape, "Labels:", y_eval.shape)

# ---- Load trained model ----
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, 2)
        self.model = base
    def forward(self, x):
        return self.model(x)

model = DeepfakeDetector()
model.load_state_dict(torch.load("visual_deepfake_model.pth"))
model.eval()

X_tensor = torch.tensor(X_eval, dtype=torch.float32)
y_tensor = torch.tensor(y_eval, dtype=torch.long)

with torch.no_grad():
    outputs = model(X_tensor)
    _, predicted = torch.max(outputs, 1)

accuracy = (predicted == y_tensor).sum().item() / len(y_tensor)
print(f"\nEVAL ACCURACY ON UNSEEN VIDEOS: {accuracy:.4f}")

correct_real = ((predicted == y_tensor) & (y_tensor == 0)).sum().item()
correct_fake = ((predicted == y_tensor) & (y_tensor == 1)).sum().item()
total_real = (y_tensor == 0).sum().item()
total_fake = (y_tensor == 1).sum().item()

print(f"Real detected correctly: {correct_real}/{total_real}")
print(f"Fake detected correctly: {correct_fake}/{total_fake}")