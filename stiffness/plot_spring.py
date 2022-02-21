import sys
sys.path.append('../utils')

import time
import matplotlib.pyplot as plt
import numpy as np
import math3d as m3d
import data
import ati
import urx

def read(*args):
    if len(args) == 0:
        name = '../data/test.csv'
    else:
        k,a,n = args
        name = '../data/stiffness_s_{:d}_{:d}_{:d}.csv'.format(k,int(a*10),n)
    d = data.read(name)
    t = np.array(d['t'])
    rz = np.array(d['rz'])
    tz = np.array(d['tz'])

    # Force bias
    to = np.nonzero(t > 1)[0][0]
    tz_offset = np.average(tz[:to])

    # Select inital point
    tz = tz-tz_offset
    i = np.nonzero(tz > 0.0001)[0][0]
    tz = tz[i:]
    rz = rz[i:]
    rz = -(rz-rz[0])

    # Fit
    k = np.linalg.lstsq(rz.reshape((-1,1)),tz,rcond=None)[0][0]
    rzp = np.arange(0,0.801,0.02)
    tzp = rzp*k

    return rz,tz,rzp,tzp,k

if __name__ == '__main__':
    plt.figure()
    for i, k in enumerate([40,60,80]):
        z,fz,zp,fzp,kp = read(k,1,1)

        c = 'C{:d}'.format(i)
        plt.plot(z,fz,'.',color=c)
        plt.plot(zp,fzp,color=c,label='{:.4f}'.format(kp))
    plt.legend()
    plt.show()
