from sklearn.neighbors import NearestNeighbors
import numpy as np
from src.ready_data import norm_data

neigh = NearestNeighbors(n_neighbors=len(norm_data))

neighborhood = []

for d in norm_data:
    dist, _ = neigh.kneighbors([d], return_distance=True)
    neighborhood.append(dist)

avg_distances = [np.mean(dist) for dist in neighborhood]
 
mean = np.mean(avg_distances)

deviations = [abs(d-mean) for d in avg_distances]