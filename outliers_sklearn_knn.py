# script calculates deviations based on standard kNN approach (Eucledian metric or Minkowski w p=2 metric)
# these deviations are plotted afterwards and it is up to the spectator or expert to choose decision boundary value d_v (0<d_<1)
# which will be used to separate data from outliers

from sklearn.neighbors import NearestNeighbors
from ready_data import norm_data
from ready_data import get_artifficial_dataset   # used for testing - call with no params a function to create dataset with 2 fake cols
import numpy as np
import matplotlib.pyplot as plt

norm_data = norm_data.T

norm_data = get_artifficial_dataset()

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

print(norm_data.iloc[-2:])

if __name__ == '__main__':
    plt.scatter([i for i in range(len(deviations))],deviations)
    plt.title("Deviations calculated from scikit-learn distances")
    #plt.grid(True)
    plt.show()