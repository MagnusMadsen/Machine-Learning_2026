import cv2
from rembg import remove
import os
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor

video_path = "Videoer/Daniel.mp4"
num_screenshots = 10
start_index = 0
output_size = (512, 512)
output_dir = "screenshots"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Cannot open video!")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_indices = sorted(random.sample(range(total_frames), num_screenshots))

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise RuntimeError("Cannot load Haar cascade")

def process_frame(frame, file_idx):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(80, 80)
    )

    if len(faces) == 0:
        return

    x, y, w, h = max(faces, key=lambda r: r[2]*r[3])

    pad_w = int(0.1 * w)
    pad_h = int(0.1 * h)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(frame.shape[1], x + w + pad_w)
    y2 = min(frame.shape[0], y + h + pad_h)

    face_crop = frame[y1:y2, x1:x2]

    face_rgba = remove(face_crop)
    alpha = face_rgba[:, :, 3]
    b, g, r = cv2.split(face_crop)
    face_png = cv2.merge([b, g, r, alpha])

    face_png = cv2.resize(face_png, output_size, interpolation=cv2.INTER_LINEAR)

    save_path = os.path.join(output_dir, f"face_{start_index + file_idx}.png")
    cv2.imwrite(save_path, face_png)

frames_to_process = []
frame_idx_set = set(frame_indices)

frame_idx = 0
while frame_idx < total_frames:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx in frame_idx_set:
        file_idx = frame_indices.index(frame_idx)
        frames_to_process.append((frame.copy(), file_idx))

    frame_idx += 1

cap.release()

with ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(lambda args: process_frame(*args), frames_to_process)

print(f"Done! {num_screenshots} face PNGs saved.")