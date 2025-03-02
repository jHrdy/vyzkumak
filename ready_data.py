import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

data = pd.read_parquet('data_BM01P1_hits.parquet')      # hity v hodoskope

def make_names(len_names):
    i = 0
    j = 0
    cnt = 0
    abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' # len = 26
    names = []
    while True:
        while j < len(abc):
            name = abc[i]+abc[j]
            names.append(name)
            cnt += 1
            j += 1
            if cnt == len_names:
                return names
        i += 1 
        j = 0 

def num_to_name(num) -> str:
    abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    first = abc[int(num / len(abc))]
    second = abc[(num % len(abc)) - 1]
    return first + second

def modify_dataset(df, names):
    
    dataset_tmp = {}
    for col in df.columns:
        dataset_tmp[names[col]] = df[col].values

    df = pd.DataFrame(dataset_tmp)
    return df

def plothist(num):
    idxs = [i for i in range(20)]
    plt.bar(idxs, data.T[num], width=0.8)
    plt.gca().set_xticks(np.arange(0, 19 + 1, 1))       # rozdelenie osy X na 20 binov
    plt.show()
    
"""
def plothist_and_save(num, deviation):
    idxs = [i for i in range(20)]
    plt.title(f'Histogram No. {num} deviation: {deviation}')
    plt.xlabel('bins')
    plt.ylabel('value')
    plt.bar(idxs, data.T[num], width=0.8)
    plt.gca().set_xticks(np.arange(0, 19 + 1, 1))       
    plt.savefig(f"plots/plot_{num}.jpg")
    plt.clf()
"""

def normalize(vec):
    top = max(vec)
    vect = []
    for i in range(len(vec)):
        vect.append(vec[i]/top)
    return vect

dataset = {}

for i in range(len(data.values)):
    dataset[i] = normalize(data.T[i])

norm_data = pd.DataFrame(data=dataset)
norm_data_copy = norm_data

names = make_names(len(data))
norm_data = modify_dataset(norm_data, names)     # v tomto formáte dataset konečne funguje (neodporúčam sa chytať ničoho, čo je vyššie)
norm_data = norm_data.T