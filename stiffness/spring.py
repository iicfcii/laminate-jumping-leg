import time
import matplotlib.pyplot as plt
import numpy as np
from utils import data
from anchor import stiffness
from template import jump

def read(s,k,a,n):
    name = './data/spring_{:d}_{:d}_{:d}_{:d}.csv'.format(s,k,int(a*10),n)
    d = data.read(name)
    t = np.array(d['t'])
    rz = np.array(d['rz'])
    tz = np.array(d['tz'])

    # Avoid jump between -pi and pi
    rz[rz < 0] += 2*np.pi

    # Force bias
    tz_offset = np.average(tz[t<3])
    tz = tz-tz_offset

    # Set dir
    dir = np.sign(tz[-1])
    tz = dir*tz
    rz = -dir*rz

    # Find contact time
    ti = t[np.nonzero(tz > 0.0002)[0][0]]

    # Select range
    tz = tz[t>ti]
    rz = rz[t>ti]
    rz = rz-rz[0]

    return rz,tz

r = 0.06
d = 0.06
if __name__ == '__main__':
    plt.figure()
    for i,k in enumerate([30]):
        c = 'C{:d}'.format(i)
        for j,a in enumerate([1]):
            for s in [1]:
                for n in [0,1,2,3]:
                    rz,tz = read(s,k,a,n)
                    plt.plot(rz*r,tz/r,'.',color=c,markersize=0.1)

            rzp = np.linspace(0,rz[-1],100)
            tzp = -jump.f_spring(rzp*r,k,a,d)*r
            plt.plot(rzp*r,tzp/r,color=c)
    plt.show()
