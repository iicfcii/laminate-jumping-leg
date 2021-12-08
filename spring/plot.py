import sys
sys.path.append('../utils')

import numpy as np
import matplotlib.pyplot as plt
import data
import process

for thickness in [15,30]:
    plt.figure()
    for i, width in enumerate([10,20,30]):
        for j, length in enumerate([25,50,75]):
            c = 'C{:d}'.format(i)
            if length == 25:
                l = '-'
                d = (None,None)
                gap = 4
            elif length == 50:
                l = '--'
                d = (10,10)
                gap = 2
            else:
                l = '--'
                d = (5,5)
                gap = 1

            ps0 = []
            tzs0 = []
            for sample in [1]:
                fs = data.read('../data/{:d}mil_{:d}mm_{:d}mm_{:d}.csv'.format(thickness,length+15,width,sample))
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
            plt.plot(pvfit,tvfit,l,color=c,dashes=d,label='k={:.3f} l={} w={}'.format(k,length,width))

    plt.xlabel('Virtual Angle [rad]')
    plt.ylabel('Virtual Torque [Nm]')
    plt.title('thickness={:d}mil'.format(thickness))
    plt.legend()
plt.show()
