#!/usr/bin/env python
"""Prepare SWaT train CSV for TSLib Dataset_Custom.

Input : Time-Series-Library/dataset/SWaT/swat_train.csv (read-only symlinked
        dataset; 1-second grid, 51 channels, first row = group names,
        timestamp column ' Timestamp' with a leading space, dd/mm/yyyy h:mm:ss).
Output: " + os.environ.get("MTP4TS_ROOT", ".") + "/data/swat/swat.csv
        (columns: date + 51 channels, 1-minute grid via mean resampling).
"""
import pandas as pd

SRC = '" + os.environ.get("MTP4TS_ROOT", ".") + "/Time-Series-Library/dataset/SWaT/swat_train.csv'
DST = '" + os.environ.get("MTP4TS_ROOT", ".") + "/data/swat/swat.csv'

df = pd.read_csv(SRC, skiprows=[0], skipinitialspace=True)
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={df.columns[0]: 'date'})
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %I:%M:%S %p')
for c in df.columns[1:]:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(axis=1, how='all')                      # trailing-comma empties + the
                                                       # non-numeric 'Normal/Attack' label
df = df.set_index('date').sort_index()
n_sec = len(df)
df = df.resample('1min').mean().dropna()
df = df.reset_index()
df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

import os
os.makedirs(os.path.dirname(DST), exist_ok=True)
df.to_csv(DST, index=False)
print('swat: {} sec-level rows -> {} minute rows, {} channels, {} -> {}'.format(
    n_sec, len(df), df.shape[1] - 1, df['date'].iloc[0], df['date'].iloc[-1]))
