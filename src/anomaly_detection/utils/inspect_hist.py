# command line application that plots histograms of the data
# !!! Currently implemented only for n_bin = 20 (thus wont work for any other types of histograms)

import matplotlib.pyplot as plt
import random
#from ready_proj_data import norm_data_2d as data

def plot_hist(data, hist_num, n_bins=20):

    if hist_num < 0:
       return False
    
    plt.figure(figsize=(10, 5))
    plt.bar(x:=range(n_bins), data[hist_num], color='blue', alpha=0.7)

    plt.xlabel("Bin")
    plt.ylabel("Value")
    #plt.title(f"Histogram plot for pt. {hist_num}")
    plt.xticks(x) 

    plt.show()
    plt.show()
    return True

def plot_original_and_reconstructed(model, full_dataset, idx=None):
    if idx is None:
        idx = random.randint(0, len(full_dataset))

    model.eval()
    pred = model(full_dataset[idx].reshape(1, 1, full_dataset[0].shape[-1])).detach()

    pred = pred.squeeze().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(len(pred)), full_dataset[idx].squeeze(), zorder=1, color='royalblue')
    axes[0].set_title(f"Original histogram (index: {idx})")
    axes[0].set_xlabel("Bin")
    axes[0].set_ylabel("Value")

    axes[1].bar(range(len(pred)), pred, zorder=1, color='royalblue')
    axes[1].set_title(f"Recreated histogram (index: {idx})")
    axes[1].set_xlabel("Bin")
    axes[1].set_ylabel("Value")
    plt.tight_layout()
    plt.show()
