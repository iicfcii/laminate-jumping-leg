import sys
sys.path.append('../utils')
sys.path.append('../anchor')

import time
import matplotlib.pyplot as plt
import numpy as np
import data
import stiffness

def read(*args):
    if len(args) == 0:
        name = '../data/test.csv'
    else:
        k,a,n = args
        name = '../data/stiffness_{:d}_{:d}_{:d}.csv'.format(k,int(a*10),n)
    d = data.read(name)
    t = np.array(d['t'])
    rz = np.array(d['rz'])
    tz = np.array(d['tz'])

    # Avoid jump between -pi and pi
    rz[rz < 0] += 2*np.pi

    # plt.plot(t,rz,'.')
    # plt.show()

    # Force bias
    to = np.nonzero(t > 1)[0][0]
    tz_offset = np.average(tz[:to])

    # Select inital point
    tz = tz-tz_offset
    i = np.nonzero(tz < -0.0005)[0][0]
    tz = -tz[i:]
    rz = rz[i:]
    rz = rz-rz[0]

    # i = np.nonzero(rz > 0.2)[0][0]
    # tz = tz[:i]
    # rz = rz[:i]

    # Fit
    k = np.linalg.lstsq(rz.reshape((-1,1)),tz,rcond=None)[0][0]

    return rz,tz,k

xs = [
    [0.010000981639302908, 0.04141110349458153, 0.04329610757653969, 0.06840294338757574, 0.010011579253171563, -0.055473045908571605],
    [0.010002499145414241, 0.023667181334153742, 0.02488843258063794, 0.041974644792054, 0.01002793807833171, -0.5988527038522609],
]
rots = [
    0.86,
    0.42,
]
r = 0.06
if __name__ == '__main__':
    for i,k in enumerate([20,50]):
        rot = rots[i]
        x = xs[i]
        c = 'C{:d}'.format(i)
        for n in [1,2,3]:
            rz,tz,kp = read(k,1,n)
            print(kp/r**2)
            plt.plot(rz,tz,'.',color=c,markersize=1)
        rzp = np.linspace(0,rot,100)
        tzp = k*r**2*rzp
        plt.plot(rzp,tzp,color=c)

        rzs,tzs = stiffness.sim(x,rot,plot=False)
        plt.plot(rzs,tzs,'--')
    plt.show()
