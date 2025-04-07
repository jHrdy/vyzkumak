import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)

#data = np.random.gamma(1,size=200)
outlier_free_data = np.random.gamma(1,size=200)
data = outlier_free_data[(outlier_free_data < 0.25) & (outlier_free_data < 0.05)]
#data = np.random.pareto(a=1.5, size=200)
#data = np.random.pareto(a=4, size=200)
#data = np.random.pareto(a=7, size=200)    

scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1,1))

dataset = data[data < 0.2]
weak_outliers = data[(data > 0.45) & (data < 0.6)]
strong_outliers = data[data > 0.65]

dataset = np.concatenate((dataset, weak_outliers, strong_outliers))
dataset = np.random.permutation(dataset)

plt.scatter(range(len(dataset)),dataset)
plt.show()