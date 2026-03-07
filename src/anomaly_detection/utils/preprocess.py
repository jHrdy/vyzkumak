import numpy as np
import torch
import torch.nn.functional as F

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
        return np.stack(X_scaled)

def resize_histograms(histogram :torch.Tensor, target_bins: int) -> torch.Tensor:
    """
    Resample histograms to a fixed number of bins.

    Args:
        histograms: Tensor shape (batch_size, bins)
        target_bins: desired number of bins

    Returns:
        Tensor shape (batch_size, target_bins)
    """
    if not isinstance(histogram, torch.Tensor):
        histogram = torch.tensor(histogram)

    x = histogram.resize(1, 1, histogram.shape[-1])

    x = F.interpolate(
        x,
        size=target_bins,
        mode="linear",
        align_corners=False
    )

    x = x.squeeze(1)

    x = x / (x.sum(dim=1, keepdim=True) + 1e-8)

    return x.squeeze()