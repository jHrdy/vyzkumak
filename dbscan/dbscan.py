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
import time
start = time.perf_counter()
dcscan = DBSCAN()
output = dcscan.fit_predict(norm_data)
end = time.perf_counter()
print(f"Runtime length: {end - start:.5f}s")
for idx, pt in enumerate(output):
    if output[idx] == 0:
        plt.scatter(idx, pt, color='green')
    else:
        plt.scatter(idx, pt, color='red')
plt.title("DBSCAN outlier Heatmap")
plt.xlabel("Epsilon")
plt.ylabel("Number of Outliers")
plt.pause(0.1)
plt.show()
outlier_cnt = []
for dist in range(1, 100, 1):
    dist /= 10
    dcscan = DBSCAN(eps=dist)
    output = dcscan.fit_predict(norm_data)
    # not the best implementation - but safe in case DBSCAN returns sth esle from 0 -1
    outlier_cnt.append(sum([1 for i in output if i == -1])) 
    
distances = [i/10 for i in range(1, 100, 1)]
import plotting_styles as styles
styles.apply_global_style()
plt.plot(distances, outlier_cnt, **styles.line_style)
plt.title("Outliers per epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Number of outliers")
plt.xticks(np.arange(0, 11, 1))
plt.show()

outlier_cnt = []
for samples in range(1, 100, 1):
    dcscan = DBSCAN(min_samples=samples)
    output = dcscan.fit_predict(norm_data)
    outlier_cnt.append(sum([1 for i in output if i == -1]))
styles.apply_global_style()
plt.plot(range(1, 100, 1), outlier_cnt, **styles.line_style)
plt.title("Outliers per minimum samples")
plt.xlabel("Minimum samples")
plt.ylabel("Outliers")
plt.show()

param_grid = {
    'eps': [eps/10 for eps in range(1, 20, 1)],
    'min_samples': [samples for samples in range(1, 100, 1)]
}
results = []
for eps in param_grid['eps']:
    for min_samples in param_grid['min_samples']:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(norm_data)
        
        # Outlieri majú label -1
        outlier_count = np.sum(labels == -1)
        
        results.append({
            'eps': eps,
            'min_samples': min_samples,
            'outliers': outlier_count
        })

for i in results:
    if i['min_samples'] == 90:
        print(i)
exit()
import seaborn as sns

df = pd.DataFrame(results)

# Heatmapa – bez gridu, vysoký kontrast
"""
for el in results:
    if el['outliers'] > 2:
        print(el) 
    elif el['outliers'] == 1:
        print(el)
"""

heatmap_data = df.pivot(index="min_samples", columns="eps", values="outliers")

plt.figure(figsize=(10, 20))

cmap = sns.color_palette("coolwarm", as_cmap=True)

ax = sns.heatmap(
    heatmap_data,
    cmap=cmap,
    annot=False,
    fmt="d",
    cbar_kws={"label": "Number of Outliers"},
    linewidths=0,
    linecolor=None
)

# Otočenie Y osi (aby malé min_samples boli dole)
ax.invert_yaxis()

# Nastavenie redších tickov
y_ticks = ax.get_yticks()
ax.set_yticks(y_ticks[::5])  # každý 5. tick
ax.set_yticklabels([int(label) for label in ax.get_yticks()])

pos_065 = heatmap_data.columns.get_indexer([0.65], method="nearest")[0]
pos_095 = heatmap_data.columns.get_indexer([1], method="nearest")[0]

plt.axvline(x=pos_065, color="white", linewidth=1)
plt.axvline(x=pos_095, color="white", linewidth=1)

# Osi a titulok
plt.ylabel("Minimum samples")
plt.xlabel("Epsilon")
plt.title("DBSCAN Outlier Count Heatmap", fontsize=14, pad=9)

plt.tight_layout()
plt.show()