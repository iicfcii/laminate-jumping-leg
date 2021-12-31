import sys
sys.path.append('../utils')

import numpy as np
import matplotlib.pyplot as plt
import data

def sample(tmil, lmm, wmm, type, rot_offset, has_gap=True):
    fs = data.read('../data/{:d}mil_{:d}mm_{:d}mm_{}.csv'.format(int(tmil/5)*5,int(lmm),wmm,type))
    t = np.array(fs['t'])
    rot0 = np.array(fs['rot'])
    tz = np.array(fs['tz'])

    gap = np.arctan2(1.5-tmil*2.54e-2/2,lmm)*2 if has_gap else 0
    rot = rot0 + (tz>0)*gap/2 - (tz<0)*gap/2 # accounts gap
    rot -= rot_offset

    t_sample = []
    t_idx = []
    t_current = 1
    for i in range(len(t)):
        if t[i] > t_current:
            t_idx.append(i)
            t_sample.append(t[i])
            t_current += 1

    r = 50
    rot_sample = [np.average(rot[i-r:i+r]) for i in t_idx]
    tz_sample = [np.average(tz[i-r:i+r]) for i in t_idx]

    return np.array(rot_sample), np.array(tz_sample)
