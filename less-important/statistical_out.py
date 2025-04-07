import pandas as pd 
import statistics as st
from ready_data import norm_data_copy
from outliers_knn import avg
import matplotlib.pyplot as plt

def idxs(iterable):
    return [i for i in range(len(iterable))]

norm_data = norm_data_copy
data = []

for i in range(len(norm_data)):
    data.append(norm_data[i].values)

means = [st.mean(d) for d in data]
vars = [st.variance(d) for d in data]

def get_outliers():
    mean = avg(means)
    devs = [abs(m - mean) for m in means]
    devs_sorted = sorted(devs)[::-1]
    return [devs.index(outlier) for outlier in devs_sorted[:5]]

print(indexes := get_outliers())

#plt.scatter()
plt.scatter(idxs(means), means)
plt.scatter(idxs(vars), vars)
plt.title('Hist of data means')
plt.grid(True)
plt.show()
