from outliers_sklearn_knn import deviations as dataset
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
#from new_way_TESTING import dataset

def idxs(iterable):
    return [i for i in range(len(iterable))]

means = KMeans(n_clusters=2, init='k-means++', random_state=42)
labels = means.fit_predict(np.array(dataset).reshape(-1, 1))

# linked to n_neighbors=2 will not work otherwise
color_map = {0: 'green', 1: 'red'}

for idx, (data, label) in enumerate(zip(dataset, labels)):
    plt.scatter(idx, data, color=color_map[label])

plt.show()  

# plotovat boundary od KMeans
# plotovat konkretne histogramy
# 2d histogramy