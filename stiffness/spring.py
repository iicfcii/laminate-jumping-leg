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
        name = '../data/spring_{:d}_{:d}_{:d}.csv'.format(k,int(a*10),n)
    d = data.read(name)
    t = np.array(d['t'])
    rz = np.array(d['rz'])
    tz = np.array(d['tz'])

    # Avoid jump between -pi and pi
    rz[rz < 0] += 2*np.pi

    # Force bias
    to = np.nonzero(t > 1)[0][0]
    tz_offset = np.average(tz[:to])

    # Select inital point
    tz = tz-tz_offset
    i = np.nonzero(tz < -0.0006)[0][0]
    tz = -tz[i:]
    rz = rz[i:]
    rz = rz-rz[0]

    # Fit
    k = np.linalg.lstsq(rz.reshape((-1,1)),tz,rcond=None)[0][0]

    # print(k/0.06**2)
    # plt.plot(rz,tz)
    # plt.show()

    return rz,tz,k

xs = [
    [0.01007617178965517, 0.07594407000664581, 0.03096829708799543, 0.09044653719214343, 0.010004714320710782, -0.17328012762043554],
    [0.01000302890652733, 0.034461108163638676, 0.016155730795981163, 0.0470899011696795, 0.010010391777558992, -0.23192544274117788],
    [0.010005283233456279, 0.025003220520924622, 0.013700848542374744, 0.037513257061323246, 0.01074210742150901, -0.4101425073901309],
]

r = 0.06
if __name__ == '__main__':
    for i,k in enumerate([20,50,80]):
        c = 'C{:d}'.format(i)
        for n in [1,2,3]:
            rz,tz,kp = read(k,1,n)
            print(kp/r**2)
            plt.plot(rz,tz,'.',color=c,markersize=1)
        rzp = np.linspace(0,rz[-1],100)
        tzp = k*r**2*rzp
        plt.plot(rzp,tzp,color=c)

        x = xs[i]
        rzs,tzs = stiffness.sim(x,rz[-1],plot=False)
        plt.plot(rzs,tzs,'--')
    plt.show()
