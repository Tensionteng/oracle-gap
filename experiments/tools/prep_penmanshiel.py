#!/usr/bin/env python
"""Prepare Penmanshiel-2016 curtailment CSV for TSLib Dataset_Custom.

Input : " + os.environ.get("MTP4TS_ROOT", ".") + "/data/censor/penmanshiel2016_processed.npz
        (14 turbines, keys Penmanshiel_{id}_{power,wind,curt,cens,avail,time,curve_x,curve_y};
        semantics from data/build_censor_dataset.py:
          power = observed 10-min power, NaN when not observed;
          wind  = wind speed, NaN when not observed;
          avail = empirical available power at this wind (q95 power curve);
          curt  = documented curtailment flag;
          cens  = EFFECTIVE censor mask (curt & clamp binding at HIGH value)).
Output: " + os.environ.get("MTP4TS_ROOT", ".") + "/data/censor/penm_fc.csv
        (date + wind, avail, curt, cens, power; power last = target).

Turbine selection: highest effective-censor rate among turbines whose longest
NaN-free segment is acceptable: WT12 (cens rate 3.29% over observed points,
6.31% inside its 4994-step clean segment, 2016-08-31 -> 2016-10-05).
NaN rule mirrors tools/prep_censor.py: longest clean segment, interior gaps of
<= 6 steps (1 h) linearly interpolated, never across real gaps.

Note: `avail` is the q95 power curve fitted on this turbine's full year, so it
uses test-period information; it is an offline-computable statistic and is kept
as an input feature (documented), not a label.
"""
import os

import numpy as np
import pandas as pd

SRC = '" + os.environ.get("MTP4TS_ROOT", ".") + "/data/censor/penmanshiel2016_processed.npz'
DST = '" + os.environ.get("MTP4TS_ROOT", ".") + "/data/censor/penm_fc.csv'
TURBINE = '12'


def longest_seg(valid):
    idx = np.flatnonzero(~valid)
    starts = np.concatenate(([0], idx + 1))
    ends = np.concatenate((idx, [len(valid)]))
    lens = ends - starts
    k = int(np.argmax(lens))
    return int(lens[k]), int(starts[k])


z = np.load(SRC)
p = z[f'Penmanshiel_{TURBINE}_power']
w = z[f'Penmanshiel_{TURBINE}_wind']
a = z[f'Penmanshiel_{TURBINE}_avail']
curt = z[f'Penmanshiel_{TURBINE}_curt'].astype(np.float32)
cens = z[f'Penmanshiel_{TURBINE}_cens'].astype(np.float32)
t = z[f'Penmanshiel_{TURBINE}_time']

valid = ~np.isnan(p) & ~np.isnan(w) & ~np.isnan(a)
L, i0 = longest_seg(valid)
seg = slice(i0, i0 + L)

df = pd.DataFrame({'date': pd.to_datetime(t[seg]).strftime('%Y-%m-%d %H:%M:%S')})
df['wind'] = w[seg]
df['avail'] = a[seg]
df['curt'] = curt[seg]
df['cens'] = cens[seg]
df['power'] = p[seg]
df[['wind', 'avail', 'power']] = df[['wind', 'avail', 'power']].interpolate(
    method='linear', limit=6, limit_area='inside')
assert not df.isna().any().any()

os.makedirs(os.path.dirname(DST), exist_ok=True)
df.to_csv(DST, index=False)
print('penm_fc (WT{}): {} rows x {} channels, cens rate {:.4f} (of {} steps), {} -> {}'.format(
    TURBINE, len(df), df.shape[1] - 1, df['cens'].mean(), int(df['cens'].sum()),
    df['date'].iloc[0], df['date'].iloc[-1]))
