import numpy as np
import torch

def drop_empty_histograms(df) -> np.ndarray:
    zero_pts = []
    for idx, data in enumerate(df):
        if data.any() == np.zeros(96).any(): 
            zero_pts.append(idx)
    
    dataset_no_outs = np.delete(df, zero_pts, axis=0)
    print(f'Dropped indexes {zero_pts}')
    return dataset_no_outs

def minmax_scale_per_sample(X):
    is_torch = isinstance(X, torch.Tensor)
    
    if is_torch:
        X_scaled = []
        for x in X:
            x_min = x.min()
            x_max = x.max()
            if x_max == x_min:
                X_scaled.append(torch.zeros_like(x))
            else:
                X_scaled.append((x - x_min) / (x_max - x_min))
        return torch.stack(X_scaled)
    
    else:  # numpy
        X_scaled = []
        for x in X:
            x_min = x.min()
            x_max = x.max()
            if x_max == x_min:
                X_scaled.append(np.zeros_like(x))
            else:
                X_scaled.append((x - x_min) / (x_max - x_min))