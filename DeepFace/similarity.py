import numpy as np
from numpy.linalg import norm

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def compare_to_set(embedding, reference_set):
    scores = [cosine_similarity(embedding, ref) for ref in reference_set]
    return max(scores)
