import sys
sys.path.append('../template')

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
import fourbar
import jump
import opt
import data

for i,s in enumerate(opt.springs[3:6]):
    cs = opt.cs
    cs['k'] = s['k']
    cs['a'] = s['a']

    # Simulation result
    # sol = jump.solve(cs)
    # t_t = sol.t
    # dy_t = sol.y[2,:]

    datum = fourbar.jump(opt.xm,s['x'],cs,plot=True)

    file_name = '../data/leg_{:d}_{:d}_full_2.csv'.format(cs['k'],int(cs['a'])*10)
    data.write(
        file_name,
        ['t','y','dy','fy'],
        [datum['t'],datum['y'],datum['dy'],datum['fy']]
    )
