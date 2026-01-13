from deepface import DeepFace
import numpy as np
import os
from similarity import compare_to_set

# --- KONFIGURATION ---
OTHERS_DIR = "data/others"
REFERENCE_PATH = "embeddings/me.npy"

MATCH_THR = 0.68
UNCERTAIN_THR = 0.55

# --- LOAD REFERENCE ---
reference_embeddings = np.load(REFERENCE_PATH)

results = []

# --- LOOP GENNEM ALLE ANDRE ---
for file in os.listdir(OTHERS_DIR):
    path = os.path.join(OTHERS_DIR, file)

    try:
        result = DeepFace.represent(
            img_path=path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True
        )

        unknown_emb = np.array(result[0]["embedding"])
        score = compare_to_set(unknown_emb, reference_embeddings)

        if score >= MATCH_THR:
            verdict = "MATCH"
        elif score >= UNCERTAIN_THR:
            verdict = "USIKKER"
        else:
            verdict = "NO MATCH"

        results.append((file, score, verdict))

    except Exception as e:
        results.append((file, None, "FEJL"))

# --- OUTPUT ---
print(f"{'FIL':30}  SCORE     RESULTAT")
print("-" * 55)

for file, score, verdict in results:
    score_str = f"{score:.3f}" if score is not None else "N/A"
    print(f"{file:30}  {score_str:7}   {verdict}")
