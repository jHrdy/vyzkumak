from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ready_data import norm_data

from sklearn.cluster import DBSCAN

dcscan = DBSCAN()
output = dcscan.fit_predict(norm_data)

for idx, pt in enumerate(output):
    if output[idx] == 0:
        plt.scatter(idx, pt, color='green')
    else:
        plt.scatter(idx, pt, color='red')

plt.pause(0.01)
plt.show()
