from sklearn.neighbors import NearestNeighbors
import numpy as np
from src.ready_data import norm_data
from sklearn.exceptions import NotFittedError

class knnOutlierDetector:
    def __init__(self):
        #self.neigh = NearestNeighbors(n_neighbors=len(data))
        self.neighborhood = []
        self.fitted = False

    def fit(self, data, n=None):
        if not n:
            self.neigh = NearestNeighbors(n_neighbors=len(data))
        else:
            self.neigh = NearestNeighbors(n_neighbors=n)

        for d in data:
            dist, _ = self.neigh.kneighbors([d], return_distance=True)
            self.neighborhood.append(dist)

        avg_distances = [np.mean(dist) for dist in self.neighborhood]
 
        mean = np.mean(avg_distances)

        self.deviations = [abs(d-mean) for d in avg_distances]
        
        self.fitted = True
    
    def predict(self):
        if self.fitted:
            return self.deviations
        else:
            raise NotFittedError(
                "Model has not been fitted yet. Call 'fit' before 'predict'."
            )
