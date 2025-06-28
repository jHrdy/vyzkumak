# data preprocessing and altering inbetween datasets
# variable 'file' may be set to values {0,1,2} to switch inbetween datasets
# artifficial dataset may be used 
# script loggs to console what data you are using but make sure to double check your imports 
# (if an artificial dataset is used, the console logs the name of the copy to which the fake columns are appended)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import os
from sklearn.preprocessing import MinMaxScaler

try:
    datafiles = os.listdir('data/bm')
except:
    #print("Error: To properly run your script make sure you are located in same directory as ready_data.py and\n" \
    #"run script as such: python subdir/your-script.py")
    #exit()
    datafiles = os.listdir('../data/bm')

file = 1
try:
    olddata = pd.read_parquet(os.path.join('data','bm','old_data_BM01P1_hits.parquet'))  # hity v hodoskope
except:
    olddata = pd.read_parquet(os.path.join('..','data','bm','old_data_BM01P1_hits.parquet'))  # hity v hodoskope
def plothist(data, num):
    idxs = [i for i in range(len(data))]
    plt.bar(idxs, data[num], width=0.8)
    plt.gca().set_xticks(np.arange(0, len(data) + 1, 1))       # rozdelenie osy X na 20 binov
    plt.show()

#data_BM01 = normalize_data(pd.read_parquet(os.path.join('..','data','data_BM01P1_hits.parquet')).T)

#data = pd.read_parquet(os.path.join('data','data_BM01P1_hits.parquet')).T
try:
    data = pd.read_parquet(os.path.join('data','bm', datafiles[file])).T
except:
    data = pd.read_parquet(os.path.join('..', 'data', 'bm', datafiles[file])).T

data_cp = data.copy()

# artifficial data currently works only for data_BM01P1_hits.parquet
# RESOLVE: norm (function) crashes when inserted data point is zeros(N) -- temporary solution implemented
#       - resolution: MinMaxScaler implemented norm function not necessary (but don't delete)

# in order not to messup tests with artifficial dataset let's not yet delete norm function
def norm(data):
    for col in range(len(data.T)):    
        top = max(data[col])        # temporary solution
        if top == 0:
            continue
        new_col = np.array([xi/top for xi in data[col]]) 
        data[col] = new_col
    return data

def get_artifficial_dataset():
    # works only for datasets with n_bins = 20
    artifficial_data = data_cp.copy()
    fake_data = pd.DataFrame({len(data.T) : np.array([0,1] + [0 for _ in range(18)]),
                            len(data.T) + 1 : np.array([0 for _ in range(20)]),
                            len(data.T) + 2 : np.array([1 for _ in range(20)]),})

    artifficial_data = norm(pd.concat((artifficial_data, fake_data), axis=1))
    print(f"You are using artifficial dataset. Last {len(fake_data.keys())} columns are fake.")
    
    return artifficial_data.T

scaler = MinMaxScaler()
norm_data = scaler.fit_transform(data)

norm_data = norm_data.T
print('Working with', datafiles[file])

if __name__ == '__main__':
    print(norm_data[0])