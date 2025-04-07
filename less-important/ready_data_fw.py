import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# BM01P1___ch
# BM01P1___t

datafiles = ['data_BM01P1_hits.parquet', 
             'data_CE01P1_hits.parquet',
             'data_CE01P1_pmt0_hits.parquet', 
             'data_CE02P1_hits.parquet', 
             'data_CE02P1_pmt0_hits.parquet']

data = pd.read_parquet('old_data_BM01P1_hits.parquet') 

scaler = MinMaxScaler()
df = data
print(df.head())

scaler = MinMaxScaler()

# Použitie scaleru na každý stĺpec samostatne
df_scaled = df.apply(lambda col: scaler.fit_transform(col.values.reshape(-1, 1)).flatten(), axis=0)

print(df_scaled.head())

print('Normalized data')
#0  0.508860  1.0  0.667306  0.293264  0.090829  0.023109  0.005026  0.001399  0.000259 -- 0
#0.520556  1.0  0.662285  0.282541  0.091446  0.020531  0.004993  0.001477  0.000204 -- 4