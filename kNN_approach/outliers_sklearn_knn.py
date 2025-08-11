# script calculates deviations based on standard kNN approach (Eucledian metric or Minkowski w p=2 metric)
# these deviations are plotted afterwards and it is up to the spectator or expert to choose decision boundary value d_v (0<d_<1)
# which will be used to separate data from outliers

from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from ready_data import norm_data
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from metrics import capped_metric, manhattan_metric, sigmoid_metric
import plotting_styles as style

#from ready_proj_data import norm_data_2d as norm_data
#from ready_data import get_artifficial_dataset   # used for testing - call with no params a function to create dataset with 2 fake cols

neigh = NearestNeighbors(n_neighbors=len(norm_data))
#neigh = NearestNeighbors(n_neighbors=len(norm_data)) 
neigh.fit(norm_data)

neighborhood = []

for d in norm_data:
    # data is returned as 2dim array => we want : from 1dim and leave out element 0 from 2nd dimension as is refers to data point itself
    # neighborhood.append(neigh.kneighbors([d], return_distance=False)[:,1:])         
    dist, _ = neigh.kneighbors([d], return_distance=True)
    neighborhood.append(dist)

avg_distances = [np.mean(dist) for dist in neighborhood]
avg_distances_copy = avg_distances.copy()

mean = np.mean(avg_distances)

deviations = [abs(d-mean) for d in avg_distances]  

if __name__ == '__main__':
    # Std when using default metric     0.09005959896056492
    # Std when using manhattan          0.17156017223782447
    print(f"Variance of the deviations: {np.std(deviations)}")
    style.apply_global_style()
    plt.scatter(range(len(deviations)), deviations, **style.scatter_style)
    #plt.axhline(y=0.4, color='seagreen', linestyle='--', linewidth=1.95)
    plt.title(f"Calculated deviations")
    plt.xlabel("Histogram index")
    plt.ylabel("Deviation")
    plt.show()
    highest_devs = sorted(dev_dict:=enumerate(deviations), key=lambda deviation: deviation[1], reverse=False)[-5:]
    print(highest_devs[::-1])
    exit()

    # following code plots outlier count per k parameter
    outliers = []
    for k in range(1,len(norm_data)):
        neigh = NearestNeighbors(n_neighbors=k) 
        neigh.fit(norm_data)

        neighborhood = []

        for d in norm_data:
            # data is returned as 2dim array => we want : from 1dim and leave out element 0 from 2nd dimension as is refers to data point itself
            # neighborhood.append(neigh.kneighbors([d], return_distance=False)[:,1:])         
            dist, _ = neigh.kneighbors([d], return_distance=True)
            neighborhood.append(dist)

        avg_distances = [np.mean(dist) for dist in neighborhood]
        avg_distances_copy = avg_distances.copy()

        mean = np.mean(avg_distances)

        deviations = [abs(d-mean) for d in avg_distances]
        outliers.append(outs := sum([1 for dev in deviations if dev > 0.12]))
        print(outs)

    plt.title("Number of outliers per k with decision border = 0.12")
    plt.xlabel('k')
    plt.ylabel('Number of outliers')
    plt.plot(range(len(outliers)), outliers)
    plt.grid(1)
    plt.show()