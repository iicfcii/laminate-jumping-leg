import sys
sys.path.append('../utils')

import numpy as np
import matplotlib.pyplot as plt
import data

def process(tmil, lmm, wmm, type, rot_offset, has_gap=True):
    fs = data.read('../data/{:d}mil_{:d}mm_{:d}mm_{}.csv'.format(int(tmil/5)*5,int(lmm),wmm,type))
    t = np.array(fs['t'])
    rot0 = np.array(fs['rot'])
    tz = np.array(fs['tz'])

    gap = np.arctan2(1.5-tmil*2.54e-2/2,lmm)*2 if has_gap else 0
    rot = rot0 + (tz>0)*gap/2 - (tz<0)*gap/2 # accounts gap
    rot -= rot_offset

    t_sample = []
    t_idx = []
    t_current = 1
    for i in range(len(t)):
        if t[i] > t_current:
            t_idx.append(i)
            t_sample.append(t[i])
            t_current += 1

    r = 50
    rot_sample = [np.average(rot[i-r:i+r]) for i in t_idx]
    tz_sample = [np.average(tz[i-r:i+r]) for i in t_idx]

    return np.array(rot_sample), np.array(tz_sample)

plt.figure()
for k, tmil in enumerate([16.5,32.5]):
    for j, wmm in enumerate([10,20,30]):
        idx = 3*k+j+1
        plt.subplot(2,3,idx)
        for i, lmm in enumerate([25,50,75]):
            t = tmil*2.54e-5
            l = lmm/1000
            w = wmm/1000

            # Experiment
            rot, tz = process(tmil,lmm,wmm,'1',np.pi/4 if k==1 else 0)
            xs = np.cos(rot)*l
            ys = np.sin(rot)*l

            # PRBM
            gamma = 0.85
            Ktheta = 2.65
            E = 18.6e9*1.1
            I = w*t**3/12
            K = gamma*Ktheta*E*I/l

            theta = tz/l*gamma*l/K
            xms = np.cos(theta)*gamma*l+(1-gamma)*l
            yms = np.sin(theta)*gamma*l

            plt.plot(xs,ys,color='C1',label='experiment')
            plt.plot(xms,yms,'o',color='C2',markersize=1,label='model')

            plt.axis('square')
            plt.title('t={:.1f}mil w={:.1f}mm'.format(tmil,wmm))
            if idx == 1 and i == 0: plt.legend()

        plt.xlim([0.015,0.085])
        plt.ylim([-0.03,0.03])
plt.show()
