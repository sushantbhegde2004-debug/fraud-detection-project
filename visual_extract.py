import cv2
import mediapipe as mp
import numpy as np
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

base_dir = r"C:\Users\91636\project 1\Celeb-DF-v2"

def extract_face_crops(video_path, max_frames=10):
    """Extract up to max_frames face crops from a video, evenly spaced."""
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
                face_crop = cv2.resize(face_crop, (224, 224))  # standard size for CNN input
                crops.append(face_crop)

    cap.release()
    return crops

# ---- Test on a small sample first: 20 real + 20 fake videos ----
real_dir = os.path.join(base_dir, "Celeb-real")
fake_dir = os.path.join(base_dir, "Celeb-synthesis")

real_files = os.listdir(real_dir)[:100]
fake_files = os.listdir(fake_dir)[:100]

X = []
y = []

print("Processing real videos...")
for i, fname in enumerate(real_files):
    path = os.path.join(real_dir, fname)
    crops = extract_face_crops(path, max_frames=3)
    for crop in crops:
        X.append(crop)
        y.append(0)  # 0 = real
    print(f"  {i+1}/{len(real_files)}: {fname} -> {len(crops)} crops")

print("Processing fake videos...")
for i, fname in enumerate(fake_files):
    path = os.path.join(fake_dir, fname)
    crops = extract_face_crops(path, max_frames=3)
    for crop in crops:
        X.append(crop)
        y.append(1)  # 1 = fake
    print(f"  {i+1}/{len(fake_files)}: {fname} -> {len(crops)} crops")

X = np.array(X)
y = np.array(y)

print("\nFinal shapes:")
print("X:", X.shape)
print("y:", y.shape)
print("Label distribution:", np.unique(y, return_counts=True))

np.save("X_visual_features.npy", X)
np.save("y_visual_labels.npy", y)
print("Saved to X_visual_features.npy and y_visual_labels.npy")