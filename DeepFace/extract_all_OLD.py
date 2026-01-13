from deepface import DeepFace
import numpy as np
import os

DATA_DIR = "data"
OUT_DIR = "embeddings"

os.makedirs(OUT_DIR, exist_ok=True)

for person in os.listdir(DATA_DIR):
    person_dir = os.path.join(DATA_DIR, person)
    if not os.path.isdir(person_dir):
        continue

    print(f"Behandler {person}")
    embeddings = []

    for file in os.listdir(person_dir):
        path = os.path.join(person_dir, file)

        result = DeepFace.represent(
            img_path=path,
            model_name="Arcface",
            detector_backend="retinaface",
            enforce_detection=True
        )

        emb = np.array(result[0]["embedding"])
        embeddings.append(emb)

    embeddings = np.array(embeddings)
    np.save(os.path.join(OUT_DIR, f"{person}.npy"), embeddings)

    print(f"  gemt {embeddings.shape[0]} embeddings")
