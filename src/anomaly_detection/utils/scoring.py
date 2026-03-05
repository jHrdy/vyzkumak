from anomaly_detection.config.paths import DATA_DIR
from anomaly_detection.utils. load_sam_data import load_dataset
import numpy as np
import torch.nn as nn
import torch

from abc import ABC, abstractmethod

class Scorer(ABC):
    @staticmethod
    def _validate(x):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}") 
        if x.dtype != torch.float32:
            raise TypeError(f"Expected dtype float32, got {x.dtype}")
    
    @staticmethod
    def _check_dimensions(x, y):
        if x.shape != y.shape:
            raise ValueError(f"Input vectors must have matching dimensions but instead got x: {x.shape}, y: {y.shape}")

    @abstractmethod
    def score(x : torch.Tensor, y : torch.Tensor):
        pass

    def __call__(self, x: torch.Tensor, y: torch.Tensor, *args) -> torch.Tensor:
        self._validate(x)
        self._validate(y)
        self._check_dimensions(x, y)

        return self.score(x, y)

# mahalanobis scoring algorithm
class VarianceScorer(Scorer):
    def __init__(self, reconstruction_errors: np.ndarray, min_variance_coeff: float = 0.01):
        """
        reconstruction_errors: numpy array shape [N, D]
            (x - x_hat)^2 computed on TRAIN set
        """

        variances = np.var(reconstruction_errors, axis=0)
        median_var = np.median(variances)

        eps = min_variance_coeff * median_var
        self.variances = torch.tensor(np.maximum(variances, eps), dtype=torch.float32)

    def score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        self._check_dimensions(x, self.variances)

        squared_error = (x - y) ** 2
        weighted_error = squared_error / self.variances

        return weighted_error.sum()
    
class QuantileScorer(Scorer):
    def __init__(self, q : float =0.9):
        self.q = q

    def score(self, x, y):
        return torch.tensor(np.quantile(np.abs(x - y), q=self.q))
        
class MaxErrorScorer(Scorer):
    def __init__(self):
        pass
    def score(self, x, y):
        return torch.tensor(np.max(np.abs(x-y)))
    
class ReconstructionConsistencyScorer(Scorer):
    def __init__(self, encoder=None, decoder=None):
        if encoder is None or decoder is None:
            raise ValueError("Encoder and Decoder must be passed to the constructor")
        
        self.enc = encoder
        self.dec = decoder
    
    def score(self, x, y=None):
        x_dim = x.shape[-1]
        z = self.enc(x.reshape(1, 1, x_dim)).detach()
        x_hat = self.dec(z).detach()
        z_twice_encoded = self.enc(x_hat).detach()
        return torch.tensor(np.linalg.norm(z - z_twice_encoded))
    
if __name__ == '__main__':
    scorer = QuantileScorer(0.97)
    vec = [5, 4, 6, 7, 1, 9]
    vec = np.array([1, 2, 4, 0])
    vec_2 = np.array([1, 2, 5, 100])
    print(np.quantile(np.abs(vec-vec_2), 0.8))

