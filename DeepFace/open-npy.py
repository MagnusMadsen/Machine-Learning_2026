import numpy as np
# Load the .npy file
data = np.load('embeddings/Daniel.npy')
# Print the loaded data
print(data)
print("Shape of the data:", data.shape)
print("Data type:", data.dtype)
# Accessing individual embeddings (if needed)
for i, embedding in enumerate(data):
    print(f"Embedding {i}: {embedding}")
# The code above loads a NumPy array from a .npy file and prints its contents, shape, and data type.
# You can modify the file path as needed to point to your specific .npy file.
# DeepFace - Facial Recognition and Analysis Library

