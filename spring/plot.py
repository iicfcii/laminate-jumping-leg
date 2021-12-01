import sys
sys.path.append('../utils')

import numpy as np
import matplotlib.pyplot as plt
import data

GAP_COUNT = int(4/0.088)
ZERO_FORCE_TH = 1e-3

fs = data.read('../data/15mil_40mm_20mm_1_2.csv')

ps0 = np.array(fs['p'])
tzs0 = np.array(fs['tz'])

ps = ps0 + (tzs0>0)*GAP_COUNT/2 - (tzs0<0)*GAP_COUNT/2 # accounts gap
non_zero_idx = np.logical_or(tzs0 > ZERO_FORCE_TH, tzs0 < -ZERO_FORCE_TH)
ps = ps[non_zero_idx]
tzs = tzs0[non_zero_idx]

plt.figure()
plt.plot(ps0,tzs0)
plt.plot(ps,tzs,'.')
plt.show()
