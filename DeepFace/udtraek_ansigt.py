from deepface import DeepFace
import numpy as np
from numpy.linalg import norm

# ME ME ME and ME
result = DeepFace.represent(
    img_path="34343.jpg", #mig
    model_name="ArcFace",
    detector_backend="retinaface",
    enforce_detection=True
)

embedding = np.array(result[0]["embedding"])

print("Embedding-længde:", embedding.shape)
print("Første 10 værdier:", embedding[:10])

np.save("me_embedding.npy", embedding)

#tilfældig person
unknown = DeepFace.represent(
    img_path="IMG_0334.jpeg",
    model_name="ArcFace",
    detector_backend="retinaface"
)

unknown_emb = np.array(unknown[0]["embedding"])

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

score = cosine_similarity(embedding, unknown_emb)
print("Similarity score:", score)
