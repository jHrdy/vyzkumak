import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

datafiles_2d = os.listdir('data/2d_hists')

data = pd.read_parquet(os.path.join('data','2d_hists', 'ce', f_name := 'x_CE01P1___tVSch.parquet'))
scaler = MinMaxScaler()
norm_data_2d = scaler.fit_transform(data)
norm_data_2d = norm_data_2d.T
print('Using 2 dimensional data.')

if __name__ == '__main__':
    print(len(norm_data_2d[0]))