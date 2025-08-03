# list of metrics that may be used to compare kNN algorithm

import numpy as np

def capped_metric(a : np.array, b : np.array) -> float:
    return np.linalg.norm(a-b) if np.linalg.norm(a-b) < 1 else 1

def sigmoid_metric(a : np.array, b : np.array):
    return 1/(1 + pow(np.e, -np.linalg.norm(a-b)))

# skusit ine metriky
def manhattan_metric(a : np.array, b : np.array) -> float:
    return np.sum(np.abs(a - b))
