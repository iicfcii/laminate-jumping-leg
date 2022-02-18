import sys
sys.path.append('../template')

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
import fourbar
import jump
import opt
import data

sos = butter(10,10,'lowpass',fs=360,output='sos')

def read(cs,n):
    d = data.read('../data/leg_{:d}_{:d}_{:d}.csv'.format(cs['k'],int(cs['a']*10),n))
    t = np.array(d['t'])
    y = np.array(d['y'])
    grf = np.array(d['grf'])

    # Select time range
    ti = np.nonzero(t > 0.51)[0][0]
    tf = np.nonzero(t > 0.7)[0][0]
    t = t[ti:tf]
    y = y[ti:tf]

    # Remove duplicate data due to slower sampling rate of mocap
    zero_idx = np.concatenate(([0],y[1:]-y[:-1])) != 0
    t = t[zero_idx]
    y = y[zero_idx]

    # Filter
    dy_raw = (y[1:]-y[:-1])/(t[1:]-t[:-1])
    t = t[1:]-t[1]
    dy = sosfiltfilt(sos,dy_raw)

    # Select end time again after filter
    tf = np.nonzero(t > 0.1)[0][0]
    t = t[:tf]
    dy_raw = dy_raw[:tf]
    dy = dy[:tf]

    return t,dy,dy_raw

lines = []
for i,s in enumerate(opt.springs[3:4]):
    cs = opt.cs
    cs['k'] = s['k']
    cs['a'] = s['a']

    # Simulation result
    # sol = jump.solve(cs)
    # t_t = sol.t
    # dy_t = sol.y[2,:]

    datum = fourbar.jump(opt.xm,s['x'],cs,plot=True)
    tf = np.nonzero(np.array(datum['t']) > 0.1)[0][0]
    t_a = datum['t'][:tf]
    dy_a = datum['dy'][:tf]

    color = 'C{:d}'.format(i)
    for n in [1,2,3]:
        t,dy,dy_raw = read(cs,n)
        ls = plt.plot(t,dy,color=color)
        if n == 1: lines.append(ls[0])
    # lines.append(plt.plot(t_t,dy_t,'--',color=color)[0])
    lines.append(plt.plot(t_a,dy_a,'-.',color=color)[0])

# type_legend = plt.legend([lines[0],lines[1]],['exp','slip'],loc='upper right')
# plt.legend([lines[0],lines[2],lines[4]],['40','60','80'],loc='lower right',title='k [N/m]')
# plt.gca().add_artist(type_legend)
plt.ylabel('dy [m/s]')
plt.xlabel('Time [s]')
plt.show()
