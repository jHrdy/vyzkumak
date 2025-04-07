from sklearn.neighbors import LocalOutlierFactor
from ready_data import norm_data
from outliers_knn import deviations
import matplotlib.pyplot as plt

def idxs(iterable):
    return [i for i in range(len(iterable))]

lof = LocalOutlierFactor(len(norm_data))
lof_fit = lof.fit_predict(norm_data)

outliers_vs_neigh = []
for i in range(len(norm_data)-1):
    lof = LocalOutlierFactor(i+1)
    lof_fit = lof.fit_predict(norm_data)
    outliers_vs_neigh.append(len([i for i in lof_fit if i == -1]))

plt.plot(idxs(outliers_vs_neigh),outliers_vs_neigh)
plt.title("Outlier count against # of neighbors (method: LOF)")
plt.xlabel("# of neighbors")
plt.ylabel("# of outliers found")
plt.grid()
plt.show()