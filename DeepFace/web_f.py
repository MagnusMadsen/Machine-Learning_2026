import cv2
import numpy as np
from deepface import DeepFace
from identify import identify_person
from input_embeddings import load_all_embeddings

# --- LOAD DATABASE ---
_db_all = load_all_embeddings("embeddings")
_db_daniel = {"Daniel": _db_all["Daniel"]}

# intern cache
_last_faces = []
_frame_count = 0


def process_frame(frame):
    """
    Tager ét BGR-frame (OpenCV)
    Returnerer annoteret BGR-frame
    """
    global _last_faces, _frame_count

    _frame_count += 1

    # Kør face detection + embedding hver 5. frame
    if _frame_count % 5 == 0:
        try:
            _last_faces = DeepFace.represent(
                img_path=frame,
                model_name="ArcFace",
                detector_backend="opencv",
                enforce_detection=False
            )
        except Exception:
            _last_faces = []

    # Tegn resultater på frame
    for face in _last_faces:
        emb = np.array(face["embedding"])
        r = face["facial_area"]

        name, score = identify_person(emb, _db_daniel)

        # Farve: Daniel = grøn, UKENDT = rød
        if name == "UKENDT":
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        x, y, w, h = r["x"], r["y"], r["w"], r["h"]

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            f"{name} ({score:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return frame
