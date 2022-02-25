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

if __name__ == '__main__':
    for i,k in enumerate([80]):
        c = 'C{:d}'.format(i)
        for n in [1,2,3]:
            rz,tz,kp = read(80,1,n)
            print(kp)
            plt.plot(rz,tz,'.',color=c,markersize=1)
        rzp = np.linspace(0,0.4,100)
        tzp = k*0.05*0.05*rzp
        plt.plot(rzp,tzp,color=c)

    # x = [0.01000070524617546, 0.022578982818760737, 0.025663408327849472, 0.04118928197560169, 0.010298609259904563, -0.006784923455315561]
    # rzs,tzs = stiffness.sim(x,0.4,plot=False)
    # plt.plot(rzs,tzs)
    plt.show()
