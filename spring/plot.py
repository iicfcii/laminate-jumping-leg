import sys
sys.path.append('../utils')

import numpy as np
import matplotlib.pyplot as plt
import data
import process
import prbm

fig, axs = plt.subplots(2,3,sharex=True,sharey=True)
for k, tmil in enumerate([16.5,32.5]):
    for j, wmm in enumerate([10,20,30]):
        idx = 3*k+j+1
        plt.subplot(2,3,idx)
        for i, lmm in enumerate([25,50,75]):
            t = tmil*2.54e-5
            l = lmm/1000
            w = wmm/1000

            # Experiment
            rot, tz = process.sample(tmil,lmm,wmm,'1',np.pi/4 if k==1 else 0)
            xe = np.cos(rot)*l
            ye = np.sin(rot)*l

            # fea
            fs = data.read('../data/{:d}mil_{:d}mm_{:d}mm_beam.csv'.format(int(tmil/5)*5,int(lmm),wmm))
            xf = np.array(fs['x'])
            yf = np.array(fs['y'])

            # PRBM
            fs = data.read('../data/{:d}mil_{:d}mm_{:d}mm_prbm.csv'.format(int(tmil/5)*5,int(lmm),wmm))
            xm = np.array(fs['x'])
            ym = np.array(fs['y'])

            plt.plot(xe,ye,color='C1',label='experiment')
            plt.plot(xm,ym,'o',color='C2',markersize=1,label='PRBM')
            plt.plot(xf,yf,'o',color='C3',markersize=1,label='FEA')

            plt.axis('square')
            plt.title('t={:.1f}mil w={:.1f}mm'.format(tmil,wmm))
            if idx == 1 and i == 0: plt.legend()
        plt.xlim([0.015,0.085])
        plt.ylim([-0.03,0.03])
plt.gcf().add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel('x [mm]')
plt.ylabel('y [mm]')
plt.show()
