# pozriet sa na projekciu
# hist tVSch fixovat biny (-8600, -8100) na ypsilonovej ose
# preprocessovat tak ze orezeme frame tj vytvori sa interval na average z meanov a okolo neho +-200
# xova os sa menit nebude tam je to v pohode

import pandas as pd
import matplotlib.pyplot as plt
import os

data = pd.read_parquet(os.path.join('data','2d_hists','ce','x_CE01P1___tVSch.parquet'))

plt.figure(figsize=(10, 5))
plt.bar(x:=range(len(data.iloc[:,2].values)), data.iloc[:,1].values, color='blue', alpha=0.7)

plt.xlabel("Bin")
plt.ylabel("Value")
plt.title(f"Histogram plot: RAW")

from ready_proj_data import norm_data_2d as norm_data

plt.figure(figsize=(10, 5))
plt.bar(x:=range(len(norm_data[2])), norm_data[2], color='blue', alpha=0.7)

plt.xlabel("Bin")
plt.ylabel("Value")
plt.title(f"Histogram plot: Normalized")
plt.show()