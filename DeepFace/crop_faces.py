import cv2
import os
from deepface import DeepFace

DATA_DIR = "data"
OUT_DIR = "cropped_faces"

os.makedirs(OUT_DIR, exist_ok=True)

for person in os.listdir(DATA_DIR):
    src_dir = os.path.join(DATA_DIR, person)
    if not os.path.isdir(src_dir):
        continue

    dst_dir = os.path.join(OUT_DIR, person)
    os.makedirs(dst_dir, exist_ok=True)

    print(f"Cropper {person}")

    for file in os.listdir(src_dir):
        path = os.path.join(src_dir, file)
        img = cv2.imread(path)
        if img is None:
            continue

        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend="retinaface",
            enforce_detection=True
        )

        face = faces[0]
        region = face["facial_area"]

        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        cropped = img[y:y+h, x:x+w]

        cv2.imwrite(os.path.join(dst_dir, file), cropped)
