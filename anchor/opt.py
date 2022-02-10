import sys
sys.path.append('../template')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, NonlinearConstraint
import fourbar
import jump

def obj_motion(x,plot=False):
    rots = np.linspace(0,-cs['dl']/cs['r'],10)+x[0]
    try:
        xs,ys = fourbar.motion(rots,x,plot=plot)
    except AssertionError:
        return 1

    xs_d = np.zeros(xs.shape)
    ys_d = (rots-x[0])*cs['r']+ys[0]

    if plot:
        plt.figure()
        plt.plot(rots,ys_d)
        plt.plot(rots,ys,'.')
        plt.plot(rots,xs_d)
        plt.plot(rots,xs,'.')
        print('range',cs['dl']/cs['r'])

    ex = np.sqrt(np.sum((xs-xs_d)**2)/xs.shape[0])
    ey = np.sqrt(np.sum((ys-ys_d)**2)/ys.shape[0])
    return ex+ey

def cb(x,convergence=0):
    print('x',x)
    print('Convergence',convergence)

def obj_stiffness(x,m,plot=False):
    try:
        datum = fourbar.stiffness(x,cs,plot=plot)
    except AssertionError:
        return 10

    x_d = np.linspace(-cs['dsm'],0,50)
    f_d = -jump.f_spring(x_d,cs)

    l = len(datum['x'])

    x1 = datum['x'][:int(l/2)]
    x1.reverse()
    x1 = np.array(x1)*cs['r']
    x2 = datum['x'][int(l/2):]
    x2 = np.array(x2)*cs['r']

    f1 = datum['f'][:int(l/2)]
    f1.reverse()
    f2 = datum['f'][int(l/2):]
    f1_i = np.interp(x_d,x1,f1)
    f2_i = np.interp(x_d,x2,f2)
    f = (f1_i+f2_i)/2/cs['r']

    e = np.sqrt(np.sum((f-f_d)**2)/f_d.shape[0])

    mass_spring = (
        (x[0]+x[2]+x[3])*fourbar.tr*fourbar.wr*fourbar.rho+
        x[1]*fourbar.tf*x[4]*fourbar.rho
    )
    em = np.abs(m-mass_spring)

    if plot:
        plt.figure()
        plt.plot(-x_d,-f_d)
        plt.plot(-x_d,-f,'--')
        # plt.plot(-np.array(datum['x'])*cs['r'],-np.array(datum['f'])/cs['r'],'--')
        print('mass error',em)

    return e+em

cs = {
    'g': 9.81,
    'mb': 0.03,
    'ml': 0.0001,
    'k': 50,
    'a': 1,
    'ds': 0.05,
    'tau': 0.109872,
    'v': 30.410616886749196,
    'dl': 0.06,
    'r': 0.05
}
sol = jump.solve(cs)
cs['dsm'] = -np.min(sol.y[1,:]) # spring travel range that needs to be matched
bounds_motion = [(-np.pi,np.pi)]+[(0.02,0.06)]*5+[(-1,1)]*1
bounds_stiffness = [(0.01,0.15)]*4+[(0.01,0.03)]+[(-1,1)]*1

if __name__ == '__main__':
    xm = None
    xs = None
    xm = [2.6935314437637747, 0.030244462243688645, 0.04668319649977162, 0.02002235749858264, 0.05998841948291793, 0.059996931859852574, 0.14061111190360398]

    # k = 50, a = 1
    xs = [0.010006806410504887, 0.04627745489917783, 0.0281204554460621, 0.06394194293073623, 0.010000796922791955, -0.1384887158574043]

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

    mass_spring = cs['mb']-0.02-np.sum(xm[1:6])*fourbar.tr*fourbar.wr*fourbar.rho
    if xs is None:
        res = differential_evolution(
            obj_stiffness,
            bounds=bounds_stiffness,
            args=(mass_spring,),
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
    print(obj_stiffness(xs,mass_spring,plot=True))
    plt.show()
