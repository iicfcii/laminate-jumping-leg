import sys
sys.path.append('../template')

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
import fourbar
import jump
import opt
import data

sos = butter(5,20,'lowpass',fs=360,output='sos')

def read(k,a,n):
    d = data.read('../data/leg_{:d}_{:d}_{:d}.csv'.format(k,int(a*10),n))
    t = np.array(d['t'])
    y = np.array(d['y'])
    grf = np.array(d['grf'])

    # Select time range
    ti = np.nonzero(t > 0.5)[0][0]
    tf = np.nonzero(t > 1.0)[0][0]
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
    ti = np.linspace(0,0.1,50)
    dy_rawi = np.interp(ti,t,dy_raw)
    dyi = np.interp(ti,t,dy)

    return ti,dyi,dy_rawi

lines = []
for i,s in enumerate(opt.springs[3:6]):
    k = s['k']
    a = s['a']

    datum = data.read('../data/leg_{:d}_{:d}_full.csv'.format(k,int(a*10)))
    tf = np.nonzero(np.array(datum['t']) > 0.1)[0][0]
    t_a = datum['t'][:tf]
    dy_a = datum['dy'][:tf]

    color = 'C{:d}'.format(i)
    dys = []
    for n in [1,2,3]:
        t,dy,dy_raw = read(k,a,n)
        dys.append(dy)
        # plt.plot(t,dy_raw,color=color)
        # ls = plt.plot(t,dy,color=color)
        # if n == 1: lines.append(ls[0])
    plt.fill_between(t, np.amax(dys,axis=0), np.amin(dys,axis=0), alpha=.5, linewidth=0)
    lines.append(plt.plot(t,np.average(dys,axis=0),'-',color=color)[0])
    lines.append(plt.plot(t_a,dy_a,'--',color=color)[0])

# type_legend = plt.legend([lines[0],lines[1],lines[2]],['exp','SLIP','full'],loc='upper right')
# plt.legend([lines[0],lines[3],lines[6]],['40','60','80'],loc='lower right',title='k [N/m]')
# plt.gca().add_artist(type_legend)
plt.ylabel('dy [m/s]')
plt.xlabel('Time [s]')
plt.show()
