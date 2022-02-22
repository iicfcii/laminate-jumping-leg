import sys
sys.path.append('../utils')
sys.path.append('../anchor')

import time
import matplotlib.pyplot as plt
import numpy as np
import data
import stiffness

def read(*args):
    if len(args) == 0:
        name = '../data/test.csv'
    else:
        k,a,n = args
        name = '../data/stiffness_s_{:d}_{:d}_{:d}.csv'.format(k,int(a*10),n)
    d = data.read(name)
    t = np.array(d['t'])
    rz = np.array(d['rz'])
    tz = np.array(d['tz'])

    # Force bias
    to = np.nonzero(t > 1)[0][0]
    tz_offset = np.average(tz[:to])

    # Select inital point
    tz = tz-tz_offset
    i = np.nonzero(tz > 0.0001)[0][0]
    tz = tz[i:]
    rz = rz[i:]
    rz = -(rz-rz[0])

    # Fit
    k = np.linalg.lstsq(rz.reshape((-1,1)),tz,rcond=None)[0][0]

    return rz,tz,k

xs = [
    [0.010012131658306514, 0.05429677864028192, 0.035094721706915084, 0.0739106305463264, 0.01000406293402808, -0.3856909524602533],
    [0.010001950482507765, 0.04039852621473746, 0.027650636112590013, 0.057829100440527456, 0.010006477628465193, -0.17417191968543333],
    [0.010002857765228523, 0.03302683572682277, 0.02177805127743359, 0.048108483937276375, 0.010006617251138208, -0.4908704598225935]
]
if __name__ == '__main__':
    plt.figure()
    for i, k in enumerate([40,60,80]):
        rz,tz,kp = read(k,1,1)
        rzp,tzp = stiffness.sim(xs[i],0.8,plot=False)

        c = 'C{:d}'.format(i)
        plt.plot(rz,tz,'.',color=c,markersize=0.5)
        plt.plot(rzp,tzp,color=c,label='{:.4f}'.format(kp))
    plt.legend()
    plt.show()
