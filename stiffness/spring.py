import time
import matplotlib.pyplot as plt
import numpy as np
from utils import data
from anchor import stiffness
from template import jump

def read(s,k,a,n):
    name = './data/spring_{:d}_{:d}_{:d}_{:d}.csv'.format(s,int(k*100),int(a*10),n)
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
    ti = t[np.nonzero(tz > 0.001)[0][0]]

    # Select range
    tz = tz[t>ti]
    rz = rz[t>ti]
    rz = rz-rz[0]

    return rz,tz

if __name__ == '__main__':
    plt.figure()
    for i,k in enumerate([0.15]):
        c = 'C{:d}'.format(i)
        for j,a in enumerate([0.5,1,2]):
            for s in [3]:
                for n in [1,2,3]:
                    rz,tz = read(s,k,a,n)
                    plt.plot(rz,tz,'.',color=c,markersize=0.5)

            rzp = np.linspace(0,jump.cs['t']/k,100)
            tzp = jump.t_spring(rzp,k,a,jump.cs['t'])
            plt.plot(rzp,tzp,color=c)
    plt.show()
