import sys
sys.path.append('../template')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, NonlinearConstraint
import fourbar
import jump

cs = {
    'g': 9.81,
    'mb': 0.03,
    'ml': 0.0001,
    'k': 30,
    'a': 1,
    'ds': 0.05,
    'tau': 0.109872,
    'v': 30.410616886749196,
    'dl': 0.06,
    'r': 0.05
}
e_max = 1
bounds = [(-np.pi,np.pi)]+[(0.02,0.06)]*5+[(-1,1)]*1

def obj(x,plot=False):
    ang = x[0]
    l = x[1:6]
    c = x[6]

    rots = np.arange(0,-cs['dl']/cs['r'],-0.1)+ang
    try:
        xs,ys = fourbar.motion(rots,l,c,plot=plot)
    except AssertionError:
        return e_max

    xs_d = np.zeros(xs.shape)
    ys_d = (rots-ang)*cs['r']+ys[0]

    if plot:
        plt.figure()
        plt.plot(rots,ys_d)
        plt.plot(rots,ys,'.')
        plt.plot(rots,xs_d)
        plt.plot(rots,xs,'.')

    ex = np.sqrt(np.sum((xs-xs_d)**2)/xs.shape[0])
    ey = np.sqrt(np.sum((ys-ys_d)**2)/ys.shape[0])
    return ex+ey

def cb(x,convergence=0):
    print('x',x)
    print('Convergence',convergence)

def obj_spring(x,xm,plot=False):
    w = x
    ang = xm[0]
    l = xm[1:6]
    c = xm[6]

    rots = np.linspace(0,-cs['dl']/cs['r'],3)+ang
    data = fourbar.spring(rots,l,c,w,plot=plot)

    if plot:
        plt.figure()
        for datum in data:
            plt.plot(datum['x'],datum['f'])

# obj_spring(
#     [0.01,0.01],
#     [2.6528534973470124, 0.030584825727719082, 0.04617399897456951, 0.020012887441194258, 0.05999632461943326, 0.059996858416849166, 0.5067485316220086],
#     plot=True
# )
# plt.show()

if __name__ == '__main__':
    x = None
    # x = [2.6528534973470124, 0.030584825727719082, 0.04617399897456951, 0.020012887441194258, 0.05999632461943326, 0.059996858416849166, 0.5067485316220086]

    if x is not None:
        obj(x,plot=True)
    else:
        res = differential_evolution(
            obj,
            bounds=bounds,
            popsize=10,
            maxiter=500,
            tol=0.01,
            callback=cb,
            workers=-1,
            polish=False,
            disp=True
        )
        print('Result', res.message)
        print('x',str(list(res.x)))
        print('Cost', res.fun)
        obj(res.x,plot=True)
