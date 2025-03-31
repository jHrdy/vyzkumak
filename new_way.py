#from outliers_sklearn_knn import deviations as dataset
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from new_way_TESTING import dataset

def idxs(iterable):
    return [i for i in range(len(iterable))]

means = KMeans(n_clusters=2, init='k-means++', random_state=42)
labels = means.fit_predict(np.array(dataset).reshape(-1, 1))

print(means.cluster_centers_)

# linked to n_clusters=2 will not work otherwise
color_map = {0: 'green', 1: 'red'}

for idx, (data, label) in enumerate(zip(dataset, labels)):
    plt.scatter(idx, data, color=color_map[label])

plt.title("Outliers separated from data")
plt.show()

"""
Nemá asi veľký zmysel plotovať decision boundary, pre jednodimenzionálne dáta.
Rozmýšľam nad ďaľšími možnosťami. Myslím, že by mohlo byť zaujímave dívať sa na 
hodnotu centroidu ako na novú mieru outlierity čím bližšie sme k 1 tým vyššia je 
pravdepodobnosť outlierity --> toto este nie je dotiahnuta myslienka ale vidim 
tam potencial.
"""

# plotovat boundary od KMeans
# 2d histogramy SKUSIT