import sys
sys.path.append('../utils')

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
import data

def read(*args):
    if len(args) == 0:
        name = '../data/test.csv'
    else:
        k,a,n = args
        name = '../data/leg_{:d}_{:d}_{:d}.csv'.format(k,int(a*10),n)
    d = data.read(name)
    t = np.array(d['t'])
    y = np.array(d['y'])
    grf = np.array(d['grf'])

    idx_y = y != None
    t_y = t[idx_y]
    y = y[idx_y]

    # fig, ax_y = plt.subplots()
    # ax_y.plot(t_y,y)
    # ax_grf = ax_y.twinx()
    # ax_grf.plot(t,grf)
    # plt.show()

    return t,grf,t_y,y

if __name__ == '__main__':
    plt.figure()
    for n in [1,2,3]:
        t,grf,t_y,y = read(80,1,n)

        c = 'C{:d}'.format(n-1)
        plt.subplot(211)
        plt.plot(t_y,y,color=c)
        plt.subplot(212)
        plt.plot(t,grf,color=c)
    plt.show()
