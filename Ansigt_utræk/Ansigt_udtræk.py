import cv2
from rembg import remove
import os
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor

script_dir = os.path.dirname(os.path.abspath(__file__))
video_paths = [
    os.path.join(script_dir, "Videoer", "Daniel.MOV"),
    os.path.join(script_dir, "Videoer", "Magnus.MOV"),
]
num_screenshots = 10
start_index = 0
output_size = (512, 512)
base_output_dir = os.path.join(script_dir, "screenshots")
os.makedirs(base_output_dir, exist_ok=True)

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise RuntimeError("Cannot load Haar cascade")

def process_frame(frame, file_idx, output_dir):
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

for video_i, video_path in enumerate(video_paths):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video with OpenCV: {video_path}. "
            f"Tip: Install FFmpeg and ensure your OpenCV build supports .MOV (H.264/HEVC)."
        )
    # Create per-video output folder (e.g., screenshots/Daniel, screenshots/Magnus)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(base_output_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames (or cannot read frame count): {video_path}")

    # Ensure we don't request more screenshots than there are frames
    n = min(num_screenshots, total_frames)
    frame_indices = sorted(random.sample(range(total_frames), n))

    frames_to_process = []
    frame_idx_set = set(frame_indices)

    frame_idx = 0
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 🔥 ROTER 90° TIL HØJRE (VIGTIG)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        if frame_idx in frame_idx_set:
            file_idx = frame_indices.index(frame_idx)
            frames_to_process.append((frame.copy(), file_idx, output_dir))

        frame_idx += 1

    cap.release()

    def _run(args):
        frame, file_idx, out_dir = args
        process_frame(frame, file_idx, out_dir)

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(_run, frames_to_process)

    print(f"Done! {n} face PNGs saved from {os.path.basename(video_path)}.")