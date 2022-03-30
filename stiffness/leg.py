import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
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
    z = z-z[0]

    # Average step
    zm = []
    fzm = []
    d = 0.0005
    zc = 0
    while zc < z[-1]+d/2:
        idx = np.logical_and(z > zc-d/2,z<zc+d/2)
        zm.append(np.mean(z[idx]))
        fzm.append(np.mean(fz[idx]))
        zc += d

    zm = np.array(zm)
    fzm = np.array(fzm)

    # Select range
    i = np.nonzero(fzm > 0.01)[0][0]-1
    zm = zm[i:]
    fzm = fzm[i:]
    zm = zm-zm[0]

    return zm,fzm

def readn(s,k,a):
    d = 0.0005

    zs = []
    fzs = []
    zmaxs = []
    for n in [1,2,3]:
        z,fz = read(s,k,a,n)
        zs.append(z)
        fzs.append(fz)
        zmaxs.append(np.amax(z))

    zmax = np.amin(zmaxs)
    zi = np.arange(0,zmax+d/2,d)

    fzis = []
    for z,fz in zip(zs,fzs):
        fzi = np.interp(zi,z,fz)
        fzis.append(fzi)
    fzi = np.mean(fzis,axis=0)

    # for z,fz in zip(zs,fzs):
    #     plt.plot(z,fz,'.')
    # plt.plot(zi,fzi)
    # plt.show()

    return zi,fzi

def fit(z,fz,kp,ap):
    d = jump.cs['ds']
    def obj(z,k,a):
        return k*d*np.power(z/d,a)

    popt, pcov = curve_fit(obj,z,fz,p0=[kp,ap])
    k,a = popt

    return k,a

r = 0.06
d = 0.06
if __name__ == '__main__':
    plt.figure()
    for i,k in enumerate([30,70]):
        c = 'C{:d}'.format(i)
        for j,a in enumerate([0.7,1,1.5]):
            for s in [1]:
                for n in [1,2,3]:
                    z,fz = read(s,k,a,n)
                    plt.plot(z,fz,'.',color=c,markersize=2)

            zp = np.linspace(0,z[-1],100)
            fzp = -jump.f_spring(zp,k,a,d)
            plt.plot(zp,fzp,color=c)
    plt.xlabel('Displacement (m)')
    plt.ylabel('Force (N)')
    plt.show()
