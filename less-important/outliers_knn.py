import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

sys.path.append(str(parent_dir))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from ready_data import data, norm_data, plothist

#norm_data = norm_data.T
def dist(ptA, ptB):
    return np.linalg.norm(ptA-ptB)

def avg(container):
    return sum(container)/len(container)

def calculate_avg_distances(test_dataset, k):

    temp_distances = []
    average_distances = []

    for ptA in test_dataset:
        for ptB in test_dataset:
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

avg_distances = calculate_avg_distances(norm_data, 64)
mean = avg(avg_distances)
deviations = [abs(d-mean) for d in avg_distances]       # list of measure of outlierness

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