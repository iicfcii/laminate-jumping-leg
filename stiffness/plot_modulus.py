import sys
sys.path.append('../utils')

import matplotlib.pyplot as plt
import numpy as np
import data

def read(n):
    l = 150
    w = 20
    h = 0.45
    A = w*h*1e-6

    d = data.read('../data/modulus_15mil_150mm_20mm_{:d}.csv'.format(n),skip=21)
    x = np.array(d['(mm)'])
    f = np.array(d['(N)'])

    # Rmove not useful data
    xf = np.nonzero(x > 1)[0][0]
    dx_raw = x[:xf]/l
    sigma_raw = f[:xf]/A

    # Select deformation range
    xi = np.nonzero(x > 0.06)[0][0] # Avoid slack region
    xf = np.nonzero(sigma_raw > 1e7)[0][0] # Max 100N is enough
    dx = dx_raw[xi:xf]
    sigma = sigma_raw[xi:xf]

    E,b = np.linalg.lstsq(
        np.concatenate((dx.reshape((-1,1)),np.ones((dx.shape[0],1))),axis=1),
        sigma,
        rcond=None
    )[0]

    return dx_raw,sigma_raw,E,b

if __name__ == '__main__':
    plt.figure()
    Es = []
    for i,n in enumerate([1,2,3]):
        dx,sigma,E,b = read(n)
        Es.append(E/1e9)
        dxp = np.linspace(dx[0],dx[-1],100)
        sigmap = dxp*E+b

        c = 'C{:d}'.format(i)
        plt.plot(dx,sigma,'.',color=c,markersize=2,markevery=5)
        plt.plot(dxp,sigmap,color=c)
    print('Youngs modulus {:.2f} GPa'.format(np.average(Es)))
    plt.xlabel('Strain')
    plt.ylabel('Stress [Pa]')
    plt.show()
