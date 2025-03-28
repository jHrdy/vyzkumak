import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import os

datafiles = os.listdir('data/bm')
file = 2
olddata = pd.read_parquet('old_data_BM01P1_hits.parquet')  # hity v hodoskope

def plothist(data, num):
    idxs = [i for i in range(len(data))]
    plt.bar(idxs, data[num], width=0.8)
    plt.gca().set_xticks(np.arange(0, len(data) + 1, 1))       # rozdelenie osy X na 20 binov
    plt.show()

#data_BM01 = normalize_data(pd.read_parquet(os.path.join('..','data','data_BM01P1_hits.parquet')).T)

#data = pd.read_parquet(os.path.join('data','data_BM01P1_hits.parquet')).T
data = pd.read_parquet(os.path.join('data','bm', datafiles[file])).T
data_cp = data.copy()

def norm(data):
    for col in range(len(data.T)):
        top = max(data[col])
        new_col = np.array([xi/top for xi in data[col]]) 
        data[col] = new_col
    return data

norm_data = norm(data_cp)
print('Working with', datafiles[file])

if __name__ == '__main__':
    pass