import sys
sys.path.append('../utils')

import numpy as np
import matplotlib.pyplot as plt
import data
import process

for thickness in [30]:
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

            k,b,(pv,tv,pvfit,tvfit) = process.k(thickness,width,length,samples=['1','2'],has_gap=True)

            plt.plot(pv,tv,'.',color=c,markersize=0.5)
            plt.plot(pvfit,tvfit,l,color=c,dashes=d,label='k={:.3f} b={:.3f} l={} w={}'.format(k,b,length,width))

    plt.xlabel('Virtual Angle [rad]')
    plt.ylabel('Virtual Torque [Nm]')
    plt.title('thickness={:d}mil'.format(thickness))
    plt.legend()
plt.show()
