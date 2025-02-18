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

    for ptA in test_dataset.values:
        for ptB in test_dataset.values:
            if not ptA is ptB:
                temp_distances.append(dist(ptA, ptB))
        temp_distances.sort()
        average_distances.append(avg(temp_distances[:k]))
        temp_distances.clear()
    return average_distances

def first_approach(data, k):

    avg_test_distances = calculate_avg_distances(data, k)
    mean_length = avg(avg_test_distances)

    return mean_length, avg_test_distances

def plot_desired_graph():
        """
        Works only for names of len = 2
        """
        print(len(norm_data))
        while True:
            graph = input('Enter desired point number: ')
            if graph.isdigit(): 
                if int(graph) < len(avg_distances) and int(graph) > 0:
                    graph = int(graph)
                    break
                else:
                    print('Invalid input. Try again.')
            elif graph == 'exit' or graph == 'e':
                return False
            else:
                print('Invalid input. Try again.')
        plothist(graph)
        return True
    
def plot_graphs_for_boundaries():
    for dec_boundary in range(0,5):
        outs = []
        dec_boundary /= 10
        for k in range(1,len(data)):
            avg_distances = calculate_avg_distances(norm_data, k)
            mean = avg(avg_distances)
            deviations = [abs(d-mean) for d in avg_distances]      
            #idxs = [i for i in range(len(deviations))]
            outs.append(sum([1 for i in deviations if i>=dec_boundary]))
        plt.plot(outs)
        plt.title(f'Number of outliers against k for dec boundary {dec_boundary}')
        plt.xlabel('k')
        plt.ylabel('# outliers')
        plt.show()

if __name__ == '__main__':
    
    avg_distances = calculate_avg_distances(norm_data, 64)
    mean = avg(avg_distances)
    deviations = [abs(d-mean) for d in avg_distances]       # list of measure of outlierness
    idxs = [i for i in range(len(deviations))]
    
    plt.scatter(idxs, deviations)
    plt.title('Distances of histograms from avg deviation')
    plt.show()
    
    n_outliers = sum([1 for i in deviations if i>=0.2])
    print(f'Found {n_outliers} outliers.')

    while True:
        if not plot_desired_graph():
            break
    
    # plot_graphs_for_boundaries()

    exit()
