import sys
sys.path.append('../utils')

import numpy as np
import matplotlib.pyplot as plt
import data
import process

plt.figure()
for i, thickness in enumerate([15,20]):
    for j, length in enumerate([25,50]):
        c = 'C{:d}'.format(i)
        if length == 25:
            l = '-'
            d = (None,None)
            gap = 4
        else:
            l = '--'
            d = (10,10)
            gap = 2

        ps0 = []
        tzs0 = []
        for sample in [1]:
            for trial in [1]:
                fs = data.read('../data/{:d}mil_{:d}mm_20mm_{:d}_{:d}.csv'.format(thickness,length+15,sample,trial))
                ps0.append(np.array(fs['p']))
                tzs0.append(np.array(fs['tz']))

        ps0 = np.concatenate(ps0)
        tzs0 = np.concatenate(tzs0)
        ps, tzs = process.remove_gap(ps0,tzs0,gap)

        # # Raw and adjusted data
        # plt.figure()
        # plt.plot(ps0,tzs0,label='raw')
        # plt.plot(ps,tzs,'.',label='adjusted')
        # plt.xlabel('Position Count (0.088deg/count)')
        # plt.ylabel('Base Torque [Nm]')
        # plt.legend()
        # plt.show()

        pv, tv = process.base2virtual(ps,tzs,length/1000)
        k = np.linalg.lstsq(pv.reshape(-1,1),tv,rcond=None)[0][0]

        pvfit = np.array([np.amin(pv),np.amax(pv)])
        tvfit = pvfit*k

        plt.plot(pv,tv,'.',color=c,markersize=0.2)
        plt.plot(pvfit,tvfit,l,color=c,dashes=d,label='k={:.3f} t={} l={}'.format(k,thickness,length))

plt.xlabel('Virtual Angle [rad]')
plt.ylabel('Virtual Torque [Nm]')
plt.legend()
plt.show()
