from pathlib import Path
import numpy as np
import os

datasets = {
    "SI" : "..\\data\\SI\\SI01U1\\SI01U1_ch",
    "FJ" : "..\\data\\FJ\\FI01X1\\FI01X1_ch",
    "PA" : "..\\data\\MWPC\\PA01U1\\PA01U1_ch"
}

def load_dataset(dataset_name):

    if dataset_name not in datasets.keys():
        raise KeyError(f"No dataset names {dataset_name} did you mean: {datasets.keys()}?")
    else:
        if not 'data' in os.listdir():
            dataset_dir = Path(datasets[dataset_name])
        else:
            dataset_dir = Path(datasets[dataset_name].lstrip("..\\"))

    print(dataset_dir)
    files = sorted(dataset_dir.glob("*.npz"))
    
    if not files:
            raise RuntimeError("No .npz files found")
    
    dataset = []
    for f in files:
        sample = np.load(f)
        dataset.append(sample['values'])
    
    return dataset