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
    'ds': 0.01,
    'tau': 0.109872,
    'v': 30.410616886749196,
    'dl': 0.06,
    'r': 0.05
}
bounds_motion = [(-np.pi,np.pi)]+[(0.02,0.06)]*5+[(-1,1)]*1
bounds_spring = [(0.01,0.03)]*2

def obj_motion(x,plot=False):
    ang = x[0]
    l = x[1:6]
    c = x[6]

    rots = np.linspace(0,-cs['dl']/cs['r'],10)+ang
    try:
        xs,ys = fourbar.motion(rots,l,c,plot=plot)
    except AssertionError:
        return 1

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
    print(rots)
    data = fourbar.spring(rots,l,c,w,cs['ds'],plot=False)

    x_d = np.linspace(-cs['ds'],0,50)
    f_d = -jump.f_spring(x_d,cs)

    e = 0
    for datum in data:
        l = len(datum['x'])
        x1 = datum['x'][:int(l/2)]
        x1.reverse()
        f1 = datum['f'][:int(l/2)]
        f1.reverse()
        x2 = datum['x'][int(l/2):]
        f2 = datum['f'][int(l/2):]
        f1_i = np.interp(x_d,x1,f1)
        f2_i = np.interp(x_d,x2,f2)
        f = (f1_i+f2_i)/2
        e += np.sqrt(np.sum((f-f_d)**2)/f_d.shape[0])

    if plot:
        plt.figure()
        plt.plot(x_d,f_d)
        for datum,rot in zip(data,rots):
            plt.plot(datum['x'],datum['f'],'--',label='{:.1f}'.format(rot))
        plt.legend()
    return e

if __name__ == '__main__':
    xm = None
    xs = None
    xm = [2.6951962388449164, 0.030257190703705427, 0.04670519207473192, 0.020045213539287253, 0.05999409087062664, 0.05997409670936284, 0.870576427719403]
    xs = [0.01,0.01]

    if xm is None:
        res = differential_evolution(
            obj_motion,
            bounds=bounds_motion,
            popsize=10,
            maxiter=500,
            tol=0.01,
            callback=cb,
            workers=-1,
            polish=False,
            disp=True
        )
        print('Result', res.message)
        print('xm',str(list(res.x)))
        print('Cost', res.fun)
        xm = res.x

    if xs is None:
        res = differential_evolution(
            obj_spring,
            bounds=bounds_spring,
            args = (xm,),
            popsize=10,
            maxiter=500,
            tol=0.01,
            callback=cb,
            workers=-1,
            polish=False,
            disp=True
        )
        print('Result', res.message)
        print('xs',str(list(res.x)))
        print('Cost', res.fun)
        xs = res.x

    print(obj_motion(xm,plot=True))
    print(obj_spring(xs,xm,plot=True))
    plt.show()
