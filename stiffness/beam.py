import sys
sys.path.append('../utils')
sys.path.append('../anchor')

import time
import matplotlib.pyplot as plt
import numpy as np
import data
from scipy.optimize import root_scalar
import stiffness

def read(i,l,w,n):
    name = '../data/beam_{:d}_{:d}_{:d}_{:d}.csv'.format(i,l,int(np.rint(w)),n)
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
    i = np.nonzero(tz < -0.0005)[0][0]
    tz = -tz[i:]
    rz = rz[i:]
    rz = rz-rz[0]

    l = l/1000
    c = 0.01 # distance between rotation center and start of flexible beam
    d = l+c# distance between rotation center and contact point

    # The actual length of the bending beam will increase a little
    # Assume that there is a rigid part at the end of the flexible beam
    # so a, the length of the shorter one of the PRBM model stay constant
    # and b, the length of the moment arm or the longer one plus rigid link,
    # can be calculated easily
    gamma = stiffness.gamma
    a = l*(1-gamma)

    theta = [] # Virtual joint angle
    tau = [] # Virtual joint torque
    for rzi,tzi in zip(rz,tz):
        b = np.sqrt((a+c)**2+d**2-2*(a+c)*d*np.cos(rzi))

        # def eq(a):
        #     return (a+c)**2+d**2-2*(a+c)*d*np.cos(rzi)-(a/(1-gamma)*gamma)**2
        # sol = root_scalar(eq,bracket=[0,l*2],method='brentq')
        # a = sol.root
        # b = a/(1-gamma)*gamma

        thetai = np.arctan2(d*np.sin(rzi),d*np.cos(rzi)-c-a)
        f = tzi/d
        f_ang = np.pi/2-(thetai-rzi)
        taui = f*np.sin(f_ang)*b

        theta.append(thetai)
        tau.append(taui)
    theta = np.array(theta)
    tau = np.array(tau)

    # Fit
    k = np.linalg.lstsq(theta.reshape((-1,1)),tau,rcond=None)[0][0]

    w = w/1000 # 9.85mm
    t = 0.00046
    I = w*t**3/12
    E = k/stiffness.gamma/stiffness.Ktheta/I*l

    thetap = np.linspace(0,theta[-1],100)
    taup = thetap*k

    # print(E)
    # plt.figure()
    # plt.plot(theta,tau)
    # plt.plot(thetap,taup)
    # plt.show()

    return E

if __name__ == '__main__':
    l = np.array([20,40,60,80,100])
    Es = []
    for s in [1,2,3]:
        E = []
        for li in l:
            Ei = []
            for n in [1,2,3]:
                Ei.append(read(s,li,9.85,3))

            E.append(np.average(Ei)/1e9)

        Es.append(E)

    p = np.polyfit(np.tile(l,3),np.array(Es).flatten(),3)
    lp = np.linspace(l[0],l[-1],50)
    Ep = np.polyval(p,lp)

    print('E (GPa) vs l(mm)',str(list(p)))

    for i,E in enumerate(Es):
        c = 'C{:d}'.format(i)
        plt.plot(l,E,'.',color=c)
    plt.plot(lp,Ep)
    plt.show()
