import sys
sys.path.append('../utils')

import numpy as np
import matplotlib.pyplot as plt
import data
import process

plt.figure()
for i, thickness in enumerate([15,20]):
    ps0 = []
    tzs0 = []
    for trial in [1,2]:
        fs = data.read('../data/{:d}mil_40mm_20mm_1_{:d}.csv'.format(thickness,trial))

        ps0.append(np.array(fs['p']))
        tzs0.append(np.array(fs['tz']))

    ps0 = np.array(ps0).flatten()
    tzs0 = np.array(tzs0).flatten()
    ps, tzs = process.remove_gap(ps0,tzs0)

    # # Raw and adjusted data
    # plt.figure()
    # plt.plot(ps0,tzs0,label='raw')
    # plt.plot(ps,tzs,'.',label='adjusted')
    # plt.xlabel('Position Count (0.088deg/count)')
    # plt.ylabel('Base Torque [Nm]')
    # plt.legend()
    # plt.show()

    pv, tv = process.base2virtual(ps,tzs)
    k = np.linalg.lstsq(pv.reshape(-1,1),tv,rcond=None)[0][0]
    tvfit = pv*k

    c = 'C{:d}'.format(i)
    plt.plot(pv,tv,'.',color=c,markersize=0.5)
    plt.plot(pv,tvfit,color=c,label='k={:.3f} t={}'.format(k,thickness))

plt.xlabel('Virtual Angle [rad]')
plt.ylabel('Virtual Torque [Nm]')
plt.legend()
plt.show()
