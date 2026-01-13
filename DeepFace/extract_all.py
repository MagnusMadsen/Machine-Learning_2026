from deepface import DeepFace
import numpy as np
import os
import cv2

DATA_DIR = "data"
OUT_DIR = "embeddings"
CROP_DIR = "cropped_faces"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

for person in os.listdir(DATA_DIR):
    person_dir = os.path.join(DATA_DIR, person)
    if not os.path.isdir(person_dir):
        continue

    print(f"Behandler {person}")
    embeddings = []

    person_crop_dir = os.path.join(CROP_DIR, person)
    os.makedirs(person_crop_dir, exist_ok=True)

    for file in os.listdir(person_dir):
        path = os.path.join(person_dir, file)

        # Load billede
        img = cv2.imread(path)
        if img is None:
            continue

        # Detect + extract face info
        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend="retinaface",
            enforce_detection=True
        )

        # Brug første ansigt (1 person pr billede)
        face = faces[0]
        region = face["facial_area"]

        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        cropped = img[y:y+h, x:x+w]

        # Gem cropped ansigt (til dokumentation)
        crop_name = f"crop_{file}"
        cv2.imwrite(os.path.join(person_crop_dir, crop_name), cropped)

        # Embedding på CROPPED face
        result = DeepFace.represent(
            img_path=cropped,
            model_name="ArcFace",      # eller Facenet512
            detector_backend="skip",   # VIGTIGT: ingen ny detection
            enforce_detection=False
        )

        emb = np.array(result[0]["embedding"])
        embeddings.append(emb)

    embeddings = np.array(embeddings)
    np.save(os.path.join(OUT_DIR, f"{person}.npy"), embeddings)

    print(f"  gemt {embeddings.shape[0]} embeddings")
