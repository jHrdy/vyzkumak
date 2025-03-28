from sklearn.neighbors import NearestNeighbors
from ready_data import norm_data
import numpy as np
import matplotlib.pyplot as plt

norm_data = norm_data.T

neigh = NearestNeighbors(n_neighbors=len(norm_data))
neigh.fit(norm_data)

neighborhood = []
for d in norm_data.iloc:
    # data is returned as 2dim array => we want : from 1dim and leave out element 0 from 2nd dimension as is refers to data point itself
    #neighborhood.append(neigh.kneighbors([d], return_distance=False)[:,1:])         
    dist, _ = neigh.kneighbors([d], return_distance=True)
    neighborhood.append(dist)

avg_distances = [np.mean(dist) for dist in neighborhood]
mean = np.mean(avg_distances)

deviations = [abs(d-mean) for d in avg_distances]

if __name__ == '__main__':
    plt.scatter([i for i in range(len(deviations))],deviations)
    plt.title("Deviations calculated from scikit-learn distances")
    #plt.grid(True)
    plt.show()