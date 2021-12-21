import sys
sys.path.append('../utils')

import numpy as np
import matplotlib.pyplot as plt
import data

ZERO_FORCE_TH = 5e-4
ROT_MID = np.pi/4

def process(type, rot_offset):
    fs = data.read('../data/{:d}mil_{:d}mm_{:d}mm_{}.csv'.format(int(tmil/5)*5,int(lmm),wmm,type))
    gap = np.arctan2(1.5-tmil*2.54e-2/2,lmm)*2
    rots0 = np.array(fs['rot'])
    tzs0 = np.array(fs['tz'])
    rots = rots0 + (tzs0>0)*gap/2 - (tzs0<0)*gap/2 # accounts gap
    non_zero_idx = np.logical_or(tzs0 > ZERO_FORCE_TH, tzs0 < -ZERO_FORCE_TH) # remove zero force
    rots = rots[non_zero_idx]
    rots -= rot_offset
    tzs = tzs0[non_zero_idx]

    return rots, tzs

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
            rots, tzs = process('1',np.pi/4 if k==1 else 0)
            xs = np.cos(rots)*l
            ys = np.sin(rots)*l

            # FEA
            rotss, tzss = process('sim_beam',0)
            xss = np.cos(rotss)*l
            yss = np.sin(rotss)*l

            # PRBM
            gamma = 0.85
            Ktheta = 2.65
            E = 18.6e9
            I = w*t**3/12
            K = gamma*Ktheta*E*I/l

            theta = tzs/l*gamma*l/K
            xms = np.cos(theta)*gamma*l+(1-gamma)*l
            yms = np.sin(theta)*gamma*l

            plt.plot(xs,ys,color='C1',markersize=1,label='experiment')
            plt.plot(xss,yss,linestyle=(0,(1,5)),color='C3',label='fea')
            plt.plot(xms,yms,linestyle=(0,(5,5)),color='C2',label='model')

            plt.axis('square')
            if idx == 1 and i == 0: plt.legend()
            if idx == 2: plt.title('tip y [mm] vs x [mm]')
        plt.xlim([0.015,0.085])
        plt.ylim([-0.03,0.03])
plt.show()
