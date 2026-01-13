from deepface import DeepFace
import numpy as np
import os
import time

IMAGE_DIR = "data/me"
OUT_FILE = "embeddings/me.npy"

files = os.listdir(IMAGE_DIR)
print(f"Starter embedding-ekstraktion ({len(files)} billeder)")
print("-" * 50)

embeddings = []

start_total = time.time()

for i, file in enumerate(files, 1):
    path = os.path.join(IMAGE_DIR, file)
    print(f"[{i}/{len(files)}] Behandler: {file}", flush=True)

    start = time.time()

    result = DeepFace.represent(
        img_path=path,
        model_name="ArcFace",
        detector_backend="retinaface",
        enforce_detection=True
    )

    emb = np.array(result[0]["embedding"])
    embeddings.append(emb)

    print(f"    færdig på {time.time() - start:.1f}s")

embeddings = np.array(embeddings)
np.save(OUT_FILE, embeddings)

print("-" * 50)
print("FÆRDIG")
print("Antal embeddings:", embeddings.shape[0])
print("Dimension:", embeddings.shape[1])
print(f"Total tid: {time.time() - start_total:.1f}s")
