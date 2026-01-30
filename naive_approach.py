import numpy as np
import matplotlib.pyplot as plt

from src.load_sam_data import load_dataset
from src.autoencoders import minmax_scale_per_sample
dataset = load_dataset("FJ")

dataset = np.asarray(dataset)
print(dataset.shape)  # očakávame (620, 96)
dataset = minmax_scale_per_sample(dataset)

mean_hist = dataset.mean(axis=0)  # shape: (96,)

distances = np.linalg.norm(dataset - mean_hist, axis=1)

print(np.where(distances > 2))

plt.figure(figsize=(10, 4))
plt.plot(distances, marker='o', linestyle='none', alpha=0.7)
plt.xlabel("Sample index")
plt.ylabel("Distance to mean histogram")
plt.title("Distance of each histogram from the mean")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.hist(distances, bins=30)
plt.xlabel("Distance to mean")
plt.ylabel("Count")
plt.title("Distribution of distances from mean histogram")
plt.tight_layout()
plt.show()