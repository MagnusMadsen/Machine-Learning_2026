import cv2
import numpy as np
from deepface import DeepFace
from similarity import compare_to_set

REFERENCE = np.load("embeddings/me.npy")

MATCH_THR = 0.68
UNCERTAIN_THR = 0.55

cap = cv2.VideoCapture(0)

frame_count = 0
last_label = "VENTER..."
last_color = (150, 150, 150)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    # kun hver 10. frame
    if frame_count % 10 == 0:
        try:
            result = DeepFace.represent(
                img_path=frame,
                model_name="ArcFace",
                detector_backend="opencv",  # hurtigere
                enforce_detection=True
            )

            emb = np.array(result[0]["embedding"])
            score = compare_to_set(emb, REFERENCE)

            if score >= MATCH_THR:
                last_label = f"MATCH ({score:.2f})"
                last_color = (0, 255, 0)
            elif score >= UNCERTAIN_THR:
                last_label = f"USIKKER ({score:.2f})"
                last_color = (0, 255, 255)
            else:
                last_label = f"NO MATCH ({score:.2f})"
                last_color = (0, 0, 255)

        except Exception:
            last_label = "INGEN ANSIGT"
            last_color = (100, 100, 100)

    cv2.putText(frame, last_label, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                last_color, 2)

    cv2.imshow("Face Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam lukket.")