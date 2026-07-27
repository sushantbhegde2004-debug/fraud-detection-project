"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Load the saved features
X = np.load("X_features.npy")
y = np.load("y_labels.npy")

print("Loaded X:", X.shape, "y:", y.shape)

# Normalize features (helps training stability)
X = (X - X.mean()) / X.std()

# Custom Dataset
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # add channel dim -> (N, 1, 40, 200)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = AudioDataset(X, y)

# Split into train (80%) and validation (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

print("Train samples:", len(train_ds), "Val samples:", len(val_ds))

# CNN model
class AudioSpoofDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # -> 16 x 20 x 100
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # -> 32 x 10 x 50
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 10 * 50, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 2 classes: real, fake
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = AudioSpoofDetector()
print(model)


import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    # ---- Training ----
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    train_acc = correct / total

    # ---- Validation ----
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == y_batch).sum().item()
            val_total += y_batch.size(0)

    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# Save the trained model
torch.save(model.state_dict(), "audio_spoof_model.pth")
print("Model saved to audio_spoof_model.pth")

"""

import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn

# ---- Paths ----
eval_protocol_path = r"C:\Users\91636\Downloads\project\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.eval.trl.txt"
eval_audio_dir = r"C:\Users\91636\Downloads\project\LA\LA\ASVspoof2019_LA_eval\flac"

# ---- Load eval protocol ----
eval_df = pd.read_csv(eval_protocol_path, sep=" ", header=None,
                       names=["speaker_id", "filename", "system_id", "null_col", "label"])

print("Eval set size:", eval_df.shape)
print(eval_df["label"].value_counts())

# ---- Feature extraction (same function as before) ----
def extract_mfcc(filename, audio_dir, max_len=200):
    path = f"{audio_dir}\\{filename}.flac"
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

# ---- Sample a balanced eval subset: 150 real + 150 fake ----
eval_sample = pd.concat([
    eval_df[eval_df["label"] == "bonafide"].sample(150, random_state=1),
    eval_df[eval_df["label"] == "spoof"].sample(150, random_state=1)
]).reset_index(drop=True)

print("Extracting eval features for", len(eval_sample), "files...")

X_eval = []
y_eval = []
for i, row in eval_sample.iterrows():
    mfcc = extract_mfcc(row["filename"], eval_audio_dir)
    X_eval.append(mfcc)
    y_eval.append(1 if row["label"] == "spoof" else 0)
    if i % 50 == 0:
        print(f"Processed {i}/{len(eval_sample)}")

X_eval = np.array(X_eval)
y_eval = np.array(y_eval)

# Normalize the same way as training (using eval's own mean/std is a simplification;
# ideally you'd reuse train's mean/std, but this works for a first pass)
X_eval = (X_eval - X_eval.mean()) / X_eval.std()

np.save("X_eval_features.npy", X_eval)
np.save("y_eval_labels.npy", y_eval)
print("Saved eval features.")

# ---- Rebuild model architecture (must match training exactly) ----
class AudioSpoofDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 10 * 50, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = AudioSpoofDetector()
model.load_state_dict(torch.load("audio_spoof_model.pth"))
model.eval()

# ---- Run inference on eval set ----
X_tensor = torch.tensor(X_eval, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y_eval, dtype=torch.long)

with torch.no_grad():
    outputs = model(X_tensor)
    _, predicted = torch.max(outputs, 1)

accuracy = (predicted == y_tensor).sum().item() / len(y_tensor)
print(f"\nEVAL SET ACCURACY (unseen attack types): {accuracy:.4f}")

# Breakdown by class
from collections import Counter
correct_real = ((predicted == y_tensor) & (y_tensor == 0)).sum().item()
correct_fake = ((predicted == y_tensor) & (y_tensor == 1)).sum().item()
total_real = (y_tensor == 0).sum().item()
total_fake = (y_tensor == 1).sum().item()

print(f"Real detected correctly: {correct_real}/{total_real}")
print(f"Fake detected correctly: {correct_fake}/{total_fake}")