import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
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
    rz = rz-rz[0]

    rzm = []
    tzm = []
    d = 0.01
    rzc = 0
    while rzc < rz[-1]+d/2:
        idx = np.logical_and(rz>rzc-d/2,rz<rzc+d/2)
        rzm.append(np.mean(rz[idx]))
        tzm.append(np.mean(tz[idx]))
        rzc += d

    rzm = np.array(rzm)
    tzm = np.array(tzm)

    # Select range
    i = np.nonzero(tzm > 1e-3)[0][0]-1
    rzm = rzm[i:]
    tzm = tzm[i:]
    rzm = rzm-rzm[0]

    # plt.plot(rzm,tzm,'.')
    # plt.show()

    return rzm,tzm

def readn(s,k,a):
    d = 0.01

    rzs = []
    tzs = []
    rzmaxs = []
    for n in [1,2,3]:
        rz,tz = read(s,k,a,n)
        rzs.append(rz)
        tzs.append(tz)
        rzmaxs.append(np.amax(rz))

    rzmax = np.amin(rzmaxs)
    rzi = np.arange(0,rzmax+d/2,d)

    tzis = []
    for rz,tz in zip(rzs,tzs):
        tzi = np.interp(rzi,rz,tz)
        tzis.append(tzi)
    tzi = np.mean(tzis,axis=0)

    # for rz,tz in zip(rzs,tzs):
    #     plt.plot(rz,tz,'.')
    # plt.plot(rzi,tzi)
    # plt.show()

    return rzi,tzi

def fit(rz,tz,kp,ap):
    def obj(rz,k,a):
        return jump.t_spring(rz,k,a,jump.cs['t'])

    popt, pcov = curve_fit(obj,rz,tz,p0=[kp,ap])
    k,a = popt

    return k,a

if __name__ == '__main__':
    plt.figure()
    for i,k in enumerate([0.15]):
        c = 'C{:d}'.format(i)
        for j,a in enumerate([2]):
            for s in [1]:
                for n in [1,2,3]:
                    rz,tz = read(s,k,a,n)
                    plt.plot(rz,tz,'.-',color=c,markersize=4)

            rzp = np.linspace(0,jump.cs['t']/k,100)
            tzp = jump.t_spring(rzp,k,a,jump.cs['t'])
            plt.plot(rzp,tzp,color=c)
    plt.show()
