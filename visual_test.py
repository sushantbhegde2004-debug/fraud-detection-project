import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

frame_count = 0
saved_count = 0

print("Recording... press 'q' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Get bounding box from all landmark points
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))

        # Add a small margin around the face
        margin = 20
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)

        face_crop = frame[y_min:y_max, x_min:x_max]

        # Show the cropped face in a separate window
        if face_crop.size > 0:
            cv2.imshow("Cropped Face", face_crop)
            saved_count += 1

    cv2.imshow("Full Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total frames: {frame_count}")
print(f"Frames with successful face crop: {saved_count}")