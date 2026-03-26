from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.exceptions import NotFittedError

class knnOutlierDetector:
    def __init__(self):
        self.neighborhood = []
        self.fitted = False
    
    def fit(self, data, n=None):
        if not n:
            self.neigh = NearestNeighbors(n_neighbors=len(data))
            self.neigh.fit(data)
        else:
            self.neigh = NearestNeighbors(n_neighbors=n)
            self.neigh.fit(data)
        self.fitted = True
    
    def predict(self, dataset):
        self.neighborhood = []
        if self.fitted:
            pass
        else:
            raise NotFittedError(
                "Model has not been fitted yet. Call 'fit' before 'predict'."
            )
        
        for d in dataset:
            dist, _ = self.neigh.kneighbors([d], return_distance=True)
            self.neighborhood.append(dist)

        avg_distances = [np.mean(dist) for dist in self.neighborhood]
 
        mean = np.mean(avg_distances)

        self.deviations = [abs(d-mean) for d in avg_distances]
        
        self.fitted = True

        return self.deviations
