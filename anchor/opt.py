import sys
sys.path.append('../template')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, NonlinearConstraint
import geom
import motion
import stiffness
import jump

a = 1.0
d = 0.06
r = 0.05
rot = 0.5
k = 80

def obj_motion(x,plot=False):
    rots = np.linspace(0,-d/r,50)+x[0]
    try:
        xs,ys = motion.sim(rots,x,plot=plot)
    except AssertionError:
        return 1

    xs_d = np.zeros(xs.shape)
    ys_d = (rots-x[0])*r+ys[0]

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

def obj_stiffness(x,m,plot=False):
    try:
        rz,tz = stiffness.sim(x,rot,plot=plot)
    except AssertionError:
        return 10

    x_d = np.linspace(0,rot*r,50)
    f_d = -jump.f_spring(x_d,k,a,d)
    f = np.interp(x_d,rz*r,tz/r)
    e = np.sqrt(np.sum((f-f_d)**2)/f_d.shape[0])

    mass_spring = (
        (x[0]+x[2]+x[3])*geom.tr*geom.wr*geom.rho+
        x[1]*geom.tf*x[4]*geom.rho
    )
    em = np.abs(m-mass_spring)

    if plot:
        plt.figure()
        plt.plot(x_d,f_d)
        plt.plot(x_d,f,'--')
        print('mass error',em)

    return e+5*em

bounds_motion = [(-np.pi,np.pi)]+[(0.02,0.06)]*5+[(-1,1)]*1
bounds_stiffness = [(0.01,0.1)]*4+[(0.01,0.02)]+[(-1,1)]*1

xm = None
xs = None
xm = [2.6781291969012333, 0.03047198019852562, 0.04640660688263967, 0.02002618942597291, 0.05998754200403933, 0.05965694209816338, 0.31866042049334675]
xs = [0.010001992894417083, 0.0222096885842388, 0.025225594479785673, 0.0405417748441995, 0.010006173038336903, -0.21230165364789066]

if __name__ == '__main__':
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

    mass_spring = 0.02-0.015-np.sum(xm[1:6])*geom.tr*geom.wr*geom.rho
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
