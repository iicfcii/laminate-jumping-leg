import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from utils import data
from anchor import stiffness
from template import jump

def read(s,k,a,n,type='spring'):
    name = './data/{}/{}_{:d}_{:d}_{:d}_{:d}.csv'.format(type,type,s,int(k*100),int(a*10),n)
    d = data.read(name)
    t = np.array(d['t'])
    rz = np.array(d['rz'])
    rz[rz < 0] += 2*np.pi # Avoid jump between -pi and pi
    f = np.array(d['tz']) if type=='spring' else np.array(d['fx'])

    # Offset
    f_offset = np.average(f[t<3])
    f = f-f_offset
    rz = rz-rz[0]

    # Set dir
    dir = np.sign(f[-1])
    f = dir*f
    dir = np.sign(rz[-1])
    rz = dir*rz

    rzm = []
    fm = []
    d = 0.01
    rzc = 0
    while rzc < rz[-1]+d/2:
        idx = np.logical_and(rz>rzc-d/2,rz<rzc+d/2)
        rzm.append(np.mean(rz[idx]))
        fm.append(np.mean(f[idx]))
        rzc += d

    rzm = np.array(rzm)
    fm = np.array(fm)

    # Select range
    th = 1e-3 if type == 'spring' else 1e-2
    i = np.nonzero(fm > th)[0][0]-1
    rzm = rzm[i:]
    fm = fm[i:]
    rzm = rzm-rzm[0]

    # plt.plot(rzm,fm,'.')
    # plt.show()

    return rzm,fm

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
    for i,k in enumerate([0.3]):
        c = 'C{:d}'.format(i)
        for j,a in enumerate([0.5,1]):
            for s in [2]:
                for n in [1,2,3]:
                    rz,f = read(s,k,a,n,type='leg')
                    plt.plot(rz,f,'.-',color=c,markersize=4)

            rzp = np.linspace(0,jump.cs['t']/k,100)
            fp = jump.t_spring(rzp,k,a,jump.cs['t'])/0.06
            plt.plot(rzp,fp,color=c)
    plt.show()
