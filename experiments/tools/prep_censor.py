#!/usr/bin/env python
"""Prepare Kelmarsh-2016 wind-farm CSV for TSLib Dataset_Custom.

Input : " + os.environ.get("MTP4TS_ROOT", ".") + "/data/censor/processed_kelmarsh2016.npz
        (keys TKelmarsh_{power,wind,setpoint,potential,curt,curt_grid,time},
        regular 10-min grid, 52416 steps).
Output: " + os.environ.get("MTP4TS_ROOT", ".") + "/data/censor/kelmarsh_fc.csv
        (date + wind, potential, curt, curt_grid, power; power last = target).

Notes (see data/censor/DATA_README.md):
  - setpoint is all-NaN in this 2016 export -> dropped;
  - curt / curt_grid are all-False in 2016 (Kelmarsh was rejected as the S6
    curtailment candidate for this reason) -> kept as 0/1 columns anyway;
  - power/wind/potential have a leading NaN block (first valid at 4842) and
    one ~30-day NaN gap; we take the longest NaN-free segment, no
    interpolation across gaps.
"""
import os

import numpy as np
import pandas as pd

SRC = '" + os.environ.get("MTP4TS_ROOT", ".") + "/data/censor/processed_kelmarsh2016.npz'
DST = '" + os.environ.get("MTP4TS_ROOT", ".") + "/data/censor/kelmarsh_fc.csv'

z = np.load(SRC)
t = z['TKelmarsh_time']
cols = {
    'wind': z['TKelmarsh_wind'],
    'potential': z['TKelmarsh_potential'],
    'curt': z['TKelmarsh_curt'].astype(np.float32),
    'curt_grid': z['TKelmarsh_curt_grid'].astype(np.float32),
    'power': z['TKelmarsh_power'],
}

valid = ~np.isnan(z['TKelmarsh_power'])
valid &= ~np.isnan(z['TKelmarsh_wind'])
valid &= ~np.isnan(z['TKelmarsh_potential'])

# longest contiguous NaN-free segment
runs = np.diff(np.flatnonzero(np.concatenate(([True], ~valid, [True]))))
starts = np.flatnonzero(np.concatenate(([True], ~valid, [True])))[::2]
segs = list(zip(starts, starts + runs[::2]))
i0, i1 = max(segs, key=lambda s: s[1] - s[0])
while i1 > i0 and not valid[i1 - 1]:      # trim trailing invalid rows
    i1 -= 1
while i0 < i1 and not valid[i0]:          # trim leading invalid rows
    i0 += 1
print('longest clean segment: {} steps ({} -> {})'.format(i1 - i0, t[i0], t[i1 - 1]))
print('dropped: {} steps before it, {} after (incl. leading NaN block and ~30-day gap)'.format(
    i0, len(t) - i1))

df = pd.DataFrame({'date': pd.to_datetime(t[i0:i1]).strftime('%Y-%m-%d %H:%M:%S')})
for k, v in cols.items():
    df[k] = v[i0:i1]
# single isolated 10-min points may sit inside the segment (off-by-one safe):
# interpolate gaps of <= 6 steps (1 hour), never across real gaps
df[['wind', 'potential', 'power']] = df[['wind', 'potential', 'power']].interpolate(
    method='linear', limit=6, limit_area='inside')
assert not df.isna().any().any()

os.makedirs(os.path.dirname(DST), exist_ok=True)
df.to_csv(DST, index=False)
print('kelmarsh_fc: {} rows x {} channels, curt rate {:.4f}, {} -> {}'.format(
    len(df), df.shape[1] - 1, df['curt'].mean(), df['date'].iloc[0], df['date'].iloc[-1]))
