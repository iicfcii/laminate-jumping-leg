import sys
sys.path.append('../utils')

import numpy as np
import matplotlib.pyplot as plt
import data
import process

for thickness in [15,30]:
    plt.figure()
    for i, width in enumerate([10,20,30]):
        for j, length in enumerate([25,37.5,50,62.5,75]):
            c = 'C{:d}'.format(i)
            k,b,(pv,tv,pvfit,tvfit) = process.k(thickness,width,length,samples=['1'],has_gap=True)
            # k,b,(pv,tv,pvfit,tvfit) = process.k(thickness,width,length,samples=['sim_beam'],has_gap=False)

            plt.plot(pv,tv,'.',color=c,markersize=0.5)
            plt.plot(pvfit,tvfit,color=c,label='k={:.3f} b={:.3f} l={} w={}'.format(k,b,length,width))

    plt.xlabel('Virtual Angle [rad]')
    plt.ylabel('Virtual Torque [Nm]')
    plt.title('thickness={:d}mil'.format(thickness))
    plt.legend()
plt.show()
