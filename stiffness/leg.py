import time
import matplotlib.pyplot as plt
import numpy as np
from utils import data
from anchor import stiffness
from template import jump

def read(s,k,a,n):
    name = './data/leg_{:d}_{:d}_{:d}_{:d}.csv'.format(s,k,int(a*10),n)
    d = data.read(name)
    t = np.array(d['t'])
    z = np.array(d['z'])
    fz = np.array(d['fz'])

    # Force bias
    fz_offset = np.average(fz[t<3])
    fz = fz-fz_offset

    # Set dir
    fz = -fz
    z = -z

    # Find contact time
    ti = t[np.nonzero(fz > 0.015)[0][0]]

    # Select range
    fz = fz[t>ti]
    z = z[t>ti]
    z = z-z[0]

    return z,fz

r = 0.06
d = 0.06
if __name__ == '__main__':
    plt.figure()
    for i,k in enumerate([30,70]):
        c = 'C{:d}'.format(i)
        for j,a in enumerate([0.7,1]):
            for s in [1]:
                for n in [1,2,3]:
                    z,fz = read(s,k,a,n)
                    plt.plot(z,fz,'.',color=c,markersize=0.5)

            zp = np.linspace(0,z[-1],100)
            fzp = -jump.f_spring(zp,k,a,d)
            plt.plot(zp,fzp,color=c)
    plt.show()
