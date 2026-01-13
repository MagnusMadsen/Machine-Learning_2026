#| Model      | MATCH |
#| ---------- | ----- |
#| ArcFace    | ~0.68 |
#| FaceNet512 | ~0.75 |

import cv2
import numpy as np
from deepface import DeepFace
from identify import identify_person
from input_embeddings import load_all_embeddings

db = load_all_embeddings("embeddings")

cap = cv2.VideoCapture(0)
frame_count = 0
last_faces = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    if frame_count % 5 == 0:
        try:
            last_faces = DeepFace.represent(
                img_path=frame,
                model_name="ArcFace",
                detector_backend="opencv",
                enforce_detection=False
            )
        except Exception:
            last_faces = []

    for face in last_faces:
        emb = np.array(face["embedding"])
        region = face["facial_area"]

        name, score = identify_person(emb, db)

        label = f"{name} ({score:.2f})"
        color = (0, 255, 0) if name != "Skyd" else (0, 0, 255)

        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Multi-Person Face ID", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
