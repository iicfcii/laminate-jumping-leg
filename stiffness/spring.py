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
    to = np.nonzero(t > 1)[0][0]
    tz_offset = np.average(tz[:to])

    # Select inital point
    tz = tz-tz_offset
    dir = np.sign(tz[-1])
    tz = dir*tz
    rz = -dir*rz
    i = np.nonzero(tz > 0.001)[0][0]
    tz = tz[i:]
    rz = rz[i:]
    rz = rz-rz[0]

    # # Fit
    # k = np.linalg.lstsq(rz.reshape((-1,1)),tz,rcond=None)[0][0]

    # print(k/0.06**2)
    # plt.plot(rz,tz)
    # plt.show()

    return rz,tz

r = 0.06
d = 0.06
if __name__ == '__main__':
    for i,k in enumerate([70]):
        for j,a in enumerate([0.7,1,1.5]):
            c = 'C{:d}'.format(j)
            for s in [1]:
                for n in [1,2,3]:
                    rz,tz = read(s,k,a,n)
                    plt.plot(rz*r,tz/r,'.',color=c,markersize=1)

            rzp = np.linspace(0,rz[-1],100)
            tzp = -jump.f_spring(rzp*r,k,a,d)*r
            plt.plot(rzp*r,tzp/r,color=c)
    plt.xlabel('Stiffness Coefficient (N/m)')
    plt.ylabel('Nonlinearity')
    plt.show()
