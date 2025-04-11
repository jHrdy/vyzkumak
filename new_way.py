# script advances knn model based on deviations with kMeans clustering to 
# single-handedly separate dataset to classes {data, outliers}
# Key-Presumption: data has to be majority of the dataset: |data| << |outliers|

# TODO this code needs a clean up !!!

from outliers_sklearn_knn import deviations as dataset
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
#from new_way_TESTING import dataset        # fake numerical datasets for testing

"""NOTE: Make sure you are using correct datasets as 
there are multiple datasets for testing

!! Dataset is imported in outliers_sklearn_knn 
(deviations are derived from it)"""

# TESTING values that are appended to deviations (in this script called dataset)
'''dataset.append(12)
dataset.append(11)
dataset.append(14)'''
dataset.append(0.95)
dataset.append(0.97)
dataset.append(0.9)
dataset.append(0.89)
dataset.append(0.93)
dataset.append(0.85)
dataset.append(0.91)

def idxs(iterable):
    return [i for i in range(len(iterable))]

means = KMeans(n_clusters=2, init='k-means++', random_state=42)
labels = means.fit_predict(np.array(dataset).reshape(-1, 1))

print(clusters := means.cluster_centers_)

# linked to n_clusters=2 will not work otherwise
color_map = {0: 'green', 1: 'red'}

outliers = []

for idx, (data, label) in enumerate(zip(dataset, labels)):
    plt.scatter(idx, data, color=color_map[label])
    if color_map[label] == 'red':
        outliers.append(idx)

decision_boundary = (clusters[1]-clusters[0])/2 

outliers_cycle = {}

n_epochs = 1

while decision_boundary > 1:
    data_wo_outliers = {}
    for idx, (pt, label) in enumerate(zip(dataset, labels)):
        if label == 0:
            data_wo_outliers[idx] = pt
        elif not pt in outliers_cycle.values():
            outliers_cycle[idx] = pt

    data_wo_outliers_list = list(data_wo_outliers.values())

    means = KMeans(n_clusters=2, init='k-means++', random_state=42)
    labels = means.fit_predict(np.array(data_wo_outliers_list).reshape(-1, 1))
    
    clusters = means.cluster_centers_
    decision_boundary = (clusters[1]-clusters[0])/2 
    #print(f'clusters: {clusters} | dec_b = {decision_boundary}')
    n_epochs += 1

print(f"Script executed in {n_epochs} epochs")
dataset_np = np.array([x for x in dataset])

outs = [dataset.index(outlier) for outlier in dataset_np[dataset > decision_boundary]]
print(f"Outliers: {outs}")

for idx, pt in enumerate(dataset):
    if pt >= decision_boundary:
        plt.scatter(idx, pt, color='red')        
    else:
        plt.scatter(idx, pt, color='green')        

plt.title("Data and outliers separated")
plt.plot(np.array([decision_boundary for _ in range(len(dataset))]), linestyle='dashed')
plt.show()

"""
Nemá asi veľký zmysel plotovať decision boundary, pre jednodimenzionálne dáta.
Rozmýšľam nad ďaľšími možnosťami. Myslím, že by mohlo byť zaujímave dívať sa na 
hodnotu centroidu ako na novú mieru outlierity čím bližšie sme k 1 tým vyššia je 
pravdepodobnosť outlierity --> toto este nie je dotiahnuta myslienka ale vidim 
tam potencial.
"""

# plotovat boundary od KMeans
# 2d histogramy SKUSIT      ===> udelat tohle 