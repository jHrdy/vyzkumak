import torch
import numpy as np
from copy import deepcopy

class HistGenerator:
    def __init__(self, n_bins : int):
        self.n_bins = n_bins
    
    def rand_hist(self):
        return torch.tensor(np.random(self.n_bins))
    
    def semi_random(self, real_sample, real_coeff=0.6, rand_coeff=0.4):
        if len(real_sample) != self.n_bins:
            raise ValueError(f"Dimension missmatch, expcted vector of dimension {self.n_bins} but instead got {len(real_sample)}")
        
        if isinstance(real_sample, torch.Tensor):
            return (real_coeff * real_sample + torch.rand(self.n_bins) * rand_coeff)/(real_coeff + rand_coeff)
        else:
            real_sample = torch.tensor(real_sample)
            return (real_coeff * real_sample + rand_coeff * torch.rand(self.n_bins) )/(real_coeff + rand_coeff)

    def artificial_real(self, *real_hists):
        return torch.tensor(np.mean(real_hists, axis=0))
    
    def zig_zag_rand(self, zero_indexes, one_indexes, real_sample):
        if len(real_sample) != self.n_bins:
            raise ValueError("Invalid input length")

        if max(zero_indexes, default=-1) >= self.n_bins or max(one_indexes, default=-1) >= self.n_bins:
            raise ValueError("Index out of range")

        result = deepcopy(real_sample)

        if zero_indexes:
            result[zero_indexes] = 0

        if one_indexes:
            result[one_indexes] = 1

        return result

# TODO: optimize this
def get_outliers(full_dataset):
    
    gen = HistGenerator(n_bins=len(full_dataset[0]))
    FOUR_OUTLIERS = [gen.semi_random(full_dataset[0]), 
                     gen.semi_random(real_coeff=0.7, rand_coeff=0.3, real_sample=full_dataset[0]),
                     gen.zig_zag_rand(zero_indexes=[20, 21, 22, 44,45, 65, 66, 67, 68], one_indexes=[], real_sample=full_dataset[0]),
                     gen.zig_zag_rand(zero_indexes=[20, 21, 22, 44, 45, 65, 66, 67, 68], one_indexes=[77, 78, 92], real_sample=full_dataset[0])]
    del gen
    return FOUR_OUTLIERS