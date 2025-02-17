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
def plothist(num):
    idxs = [i for i in range(20)]
    plt.bar(idxs, data.T[num], width=0.8)
    plt.gca().set_xticks(np.arange(0, 19 + 1, 1))
    plt.show()

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
norm_data = norm_data.T

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

def first_approach(data, k):

    avg_test_distances = calculate_avg_distances(data, k)
    #print(f'avg test dist: {avg_test_distances} and length {len(avg_test_distances)}')
    mean_length = avg(avg_test_distances)

    outlier_cnt = 0

    """    for i in range(len(deviations)):
        if deviations[i] >= decision_boundary: #avg_test_distances[i] >= decision_boundary:                    # !!! ZDE VYMYSLET !!!
            outlier_cnt += 1
            outs.append(outlier_cnt)
    #print(f'Number of outliers {outlier_cnt} (1st approach)')"""
    return mean_length, avg_test_distances, outlier_cnt

if __name__ == '__main__':
    
    """
    Overview:
        - Measure of 'outlierness': 
            - Distance ratio is calculated as 'distance' between a average dist ratio and a ratio of average distance of each point 
            to average distance of in entire dataset
            - point = histogram
            - measure = avg(dist_ratio) - dist_ratio/avg(distances)
        - Graph shows Distances of histograms from avg deviation distances cummulate near a certain value 
        any distant values may be considered outliers
    """
    
    avg_distances = calculate_avg_distances(norm_data, 64)

    mean = avg(avg_distances)
    #norm_data['Dist ratio'] = avg_distances/mean  

    # outlier dependency on deviance from dist ratio
    # avg_deviation = avg(norm_data['Dist ratio'])
    deviations = [abs(d-mean) for d in avg_distances]       # list of measure of outlierness
    idxs = [i for i in range(len(deviations))]
    
    plt.scatter(idxs, deviations)
    plt.title('Distances of histograms from avg deviation')
    plt.show()
    
    n_outliers = sum([1 for i in deviations if i>=0.2])
    print(f'Found {n_outliers} outliers.')

    def plot_desired_graph():
        """
        Works only for names of len = 2
        """
        print(len(norm_data))
        while True:
            graph = input('Enter desired point number: ')
            if graph.isdigit(): 
                if int(graph) <= len(avg_distances) and int(graph) > 0:
                    graph = int(graph)
                    break
                else:
                    print('Invalid input. Try again.')
            elif graph == 'exit' or graph == 'e':
                return -1
            else:
                print('Invalid input. Try again.')
        plothist(graph)
        return 0

    while True:
        if plot_desired_graph() == -1:
            break
        

    def plot_graphs_for_boundaries():
        for dec_boundary in range(0,5):
            outs = []
            dec_boundary /= 10
            for k in range(1,len(data)):
                avg_distances = calculate_avg_distances(norm_data, k)
                mean = avg(avg_distances)
                deviations = [abs(d-mean) for d in avg_distances]       # list of measure of outlierness
                idxs = [i for i in range(len(deviations))]
                outs.append(sum([1 for i in deviations if i>=dec_boundary]))
            plt.plot(outs)
            plt.title(f'Number of outliers against k for dec boundary {dec_boundary}')
            plt.xlabel('k')
            plt.ylabel('# outliers')
            plt.show()

    exit()
