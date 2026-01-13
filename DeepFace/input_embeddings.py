import numpy as np
import os

def load_all_embeddings(folder="embeddings"):
    database = {}

    for file in os.listdir(folder):
        if file.endswith(".npy"):
            name = file.replace(".npy", "")
            database[name] = np.load(os.path.join(folder, file))

    return database
