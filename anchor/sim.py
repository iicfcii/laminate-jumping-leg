import sys, os
sys.path.append('../template')
sys.path.append('../utils')

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
import fourbar
import jump
import opt
import data

cwd = os.getcwd()
for i,s in enumerate(opt.springs[3:6]):
    cs = opt.cs
    cs['k'] = s['k']
    cs['a'] = s['a']

    # Simulation result
    # sol = jump.solve(cs)
    # t_t = sol.t
    # dy_t = sol.y[2,:]

    datum = fourbar.jump(opt.xm,s['x'],cs,plot=True)

    # Note: cwd seems to change after chrono vis in Mac
    file_name = os.path.join(
        cwd,
        '../data/leg_{:d}_{:d}_full.csv'.format(cs['k'],int(cs['a'])*10)
    )
    data.write(
        file_name,
        list(datum.keys()),
        list(datum.values())
    )
