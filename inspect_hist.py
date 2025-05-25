# command line application that plots histograms of the data
# !!! Currently implemented only for n_bin = 20 (thus wont work for any other types of histograms)

import matplotlib.pyplot as plt
from ready_data import norm_data, data
from ready_proj_data import norm_data_2d as data

def plot_hist(hist_num):

    if hist_num < 0:
       return False
    
    plt.figure(figsize=(10, 5))
    plt.bar(x:=range(20), data[hist_num], color='blue', alpha=0.7)

    plt.xlabel("Bin")
    plt.ylabel("Value")
    plt.title(f"Histogram plot for pt. {hist_num}")
    plt.xticks(x) 

    plt.show()
    plt.show()
    return True

if __name__ == "__main__":
    print(f'Keep entering integers in range 0 to {len(norm_data.T)}, to exit input negative integer')

    while True:
       if plot_hist(int(input("Type number of desired histogram: "))):
           continue
       else:
           break
