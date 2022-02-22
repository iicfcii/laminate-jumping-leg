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
        name = '../data/stiffness_{:d}_{:d}_{:d}.csv'.format(k,int(a*10),n)
    d = data.read(name)
    t = np.array(d['t'])
    z = np.array(d['z'])
    fz = np.array(d['fz'])

    # Force bias
    to = np.nonzero(t > 1)[0][0]
    fz_offset = np.average(fz[:to])

    # Select inital point
    fz = -(fz-fz_offset)
    i = np.nonzero(fz > 0.005)[0][0]
    fz = fz[i:]
    z = z[i:]
    z = -(z-z[0])

    # Fit
    k = np.linalg.lstsq(z.reshape((-1,1)),fz,rcond=None)[0][0]
    zp = np.arange(0,0.04,0.001)
    fzp = zp*k

    return z,fz,zp,fzp,k

if __name__ == '__main__':
    plt.figure()
    for i, k in enumerate([40,60,80]):
        z,fz,zp,fzp,kp = read(k,1,1)

        c = 'C{:d}'.format(i)
        plt.plot(z,fz,'.',color=c)
        plt.plot(zp,fzp,color=c,label='{:.1f}'.format(kp))
    plt.legend()
    plt.show()
