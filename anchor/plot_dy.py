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

lines = []
for i,s in enumerate(opt.springs[3:6]):
    cs = opt.cs
    cs['k'] = s['k']
    cs['a'] = s['a']

    d = data.read('../data/leg_{:d}_{:d}_3.csv'.format(cs['k'],int(cs['a']*10)))
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

    # Simulation result
    sol = jump.solve(cs)
    t_t = sol.t
    dy_t = sol.y[2,:]

    datum = fourbar.jump(opt.xm,s['x'],cs,plot=False)
    tf = np.nonzero(np.array(datum['t']) > t[-1])[0][0]
    t_a = datum['t'][:tf]
    dy_a = datum['dy'][:tf]

    color = 'C{:d}'.format(i+1)
    # plt.plot(t,dy_raw,color=color)
    lines.append(plt.plot(t,dy,color=color)[0])
    lines.append(plt.plot(t_t,dy_t,'--',color=color)[0])
    lines.append(plt.plot(t_a,dy_a,'-.',color=color)[0])

type_legend = plt.legend([lines[0],lines[1],lines[2]],['exp','template','anchor'],loc='upper right')
plt.legend([lines[0],lines[3],lines[6]],['40','60','80'],loc='lower right',title='k [N/m]')
plt.gca().add_artist(type_legend)
plt.ylabel('dy [m/s]')
plt.xlabel('Time [s]')
plt.show()
