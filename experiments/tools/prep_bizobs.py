#!/usr/bin/env python
"""Export bizitobs_application (GIFT-Eval, 1 item x 2 channels x 8834 steps,
10-second grid) as a TSLib Dataset_Custom wide CSV with an OT column.

The stock gapbench/adapter.py load_collection() rejects multivariate targets;
this is the 2-channel special case done directly.

Output: gapbench/csv/bizitobs_application_10S.csv (date + ch0 + OT).
"""
import os

import numpy as np
import pandas as pd
import datasets

SRC = '" + os.environ.get("GIFT_ROOT", "data/gifteval") + "/bizitobs_application/data-00000-of-00001.arrow'
DST = '" + os.environ.get("MTP4TS_ROOT", ".") + "/gapbench/csv/bizitobs_application_10S.csv'

d = datasets.Dataset.from_file(SRC)
r = d[0]
t = np.asarray(r['target'], dtype=np.float64)          # [2, 8834]
assert t.shape[0] == 2 and not np.isnan(t).any()
dates = pd.date_range(pd.Timestamp(r['start']), periods=t.shape[1], freq='10s')
df = pd.DataFrame({'date': dates.strftime('%Y-%m-%d %H:%M:%S'),
                   'ch0': t[0], 'OT': t[1]})
os.makedirs(os.path.dirname(DST), exist_ok=True)
df.to_csv(DST, index=False)
print('bizitobs_application -> {}: {} rows x 2 channels, 10s grid, {} -> {}'.format(
    DST, len(df), df['date'].iloc[0], df['date'].iloc[-1]))
