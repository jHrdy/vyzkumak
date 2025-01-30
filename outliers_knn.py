import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

data = pd.read_parquet('data_BM01P1_hits.parquet')      # hity v hodoskope

def make_names(len_names):
    
    abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' # len = 26
    names = []
    j = 0

    for i in range(len_names):
        names.append(abc[j % len(abc)] + abc[i % len(abc)])
        if i % 26 == 0 and i != 0:
            j += 1
            if j > 25:
                print('No more names left.')
                break
    return names

def modify_dataset(df, names):
    
    dataset_tmp = {}
    for col in df.columns:
        dataset_tmp[names[col]] = df[col].values

    df = pd.DataFrame(dataset_tmp)
    return df

"""
Code verview:
    - data normalization
    - hepling functions: dist, avg
    - IMPORTANT: calculate_avg_distances
                    - calculates distances between every pair of pts
                    - distances are then sorted and alg takes an average of 'k' lowest values
                    - this way every point gets assigned a value representing it's avg dist from neighbor
                    - function returns a list where ith element is the mentioned avg dist of i-th data point

    - IMPORTANT: first_approach    
                    - this approach calculated avg dist by averaging the distance from the entire dataset
                        - means - N.o. neighbors: k = len(dataset)
                    - if this avg dist exceeds (currently fixed value) 10  
    
    - second approach will iterate over values of 'k' figuring which value performs best 
"""

def normalize(vec):
    top = max(vec)
    for i in range(len(vec)):
        vec[i] = vec[i]/top
    return vec

dataset = {}

for i in range(len(data.values)):
    dataset[i] = normalize(data.T[i])

norm_data = pd.DataFrame(data=dataset)

names = make_names(len(data))
norm_data = modify_dataset(norm_data, names)     # v tomto formáte dataset konečne funguje (neodporúčam sa chytať ničoho, čo je vyššie)

def dist(ptA, ptB):
    return np.linalg.norm(ptA-ptB)

def avg(container):
    return sum(container)/len(container)

def calculate_avg_distances(test_dataset, k):

    temp_distances = []
    average_distances = []
    #distances = []

    for ptA in test_dataset.values:
        for ptB in test_dataset.values:
            if not ptA is ptB:
                temp_distances.append(dist(ptA, ptB))
                #distances.append(dist(ptA, ptB))
        temp_distances.sort()
        average_distances.append(avg(temp_distances[:k]))
        temp_distances.clear()
    return average_distances

def first_approach(data):

    avg_test_distances = calculate_avg_distances(data, len(data))
    mean_length = avg(avg_test_distances)

    outlier_cnt = 0
    outs = []
    for i in range(len(avg_test_distances)):
        if avg_test_distances[i] >= 10.5:                    # !!! ZDE VYMYSLET !!!
            outlier_cnt += 1
            outs.append(outlier_cnt)
    print(f'Number of outliers {outlier_cnt} (1st approach)')

    plt.hist(avg_test_distances)
    plt.xlabel('Average distances inbetween points')
    return mean_length, avg_test_distances

if __name__ == '__main__':

    mean, avg_distances = first_approach(norm_data)
    
    norm_data['Dist ratio'] = avg_distances/mean
    
    pprint(norm_data.head())
    