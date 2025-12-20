from pathlib import Path
import numpy as np

def load_dataset(data_dir):
    #data_dir = "data\\FJ\\FI01X1\\FI01X1_ch"

    dataset_dir = Path(data_dir)
    files = sorted(dataset_dir.glob("*.npz"))
    if not files:
        raise RuntimeError("No .npz files found")
    
    dataset = []
    for f in files:
        sample = np.load(f)
        dataset.append(sample['values'])
    
    return dataset