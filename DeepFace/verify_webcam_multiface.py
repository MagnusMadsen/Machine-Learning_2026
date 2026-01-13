import cv2
import numpy as np
from deepface import DeepFace
from similarity import compare_to_set

REFERENCE = np.load("embeddings/me.npy")

MATCH_THR = 0.68
UNCERTAIN_THR = 0.55

cap = cv2.VideoCapture(0)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    try:
        # kun hver 5. frame for hastighed
        if frame_count % 5 == 0:
            faces = DeepFace.represent(
                img_path=frame,
                model_name="ArcFace",
                detector_backend="opencv",
                enforce_detection=False
            )

            last_faces = faces

        # tegn seneste resultater
        if "last_faces" in locals():
            for face in last_faces:
                emb = np.array(face["embedding"])
                region = face["facial_area"]

                score = compare_to_set(emb, REFERENCE)

                if score >= MATCH_THR:
                    label = f"MATCH ({score:.2f})"
                    color = (0, 255, 0)
                elif score >= UNCERTAIN_THR:
                    label = f"USIKKER ({score:.2f})"
                    color = (0, 255, 255)
                else:
                    label = f"Skyd dem! ({score:.2f})"
                    color = (0, 0, 255)

                x, y, w, h = region["x"], region["y"], region["w"], region["h"]

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

    except Exception:
        pass

    cv2.imshow("Face Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows() 
