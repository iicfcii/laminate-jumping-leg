import sys
sys.path.append('../template')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, NonlinearConstraint
import geom
import motion
import stiffness
import jump

def obj_motion(x,plot=False):
    rots = np.linspace(0,-cs['dl']/cs['r'],50)+x[0]
    try:
        xs,ys = motion.sim(rots,x,plot=plot)
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
        rz,tz = stiffness.sim(x,cs['dsm']/cs['r'],plot=plot)
    except AssertionError:
        return 10

    x_d = np.linspace(0,cs['dsm'],50)
    f_d = -jump.f_spring(x_d,cs)
    f = np.interp(x_d,rz*cs['r'],tz/cs['r'])
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

cs = {
    'g': 9.81,
    'mb': 0.025,
    'ml': 0.0001,
    'k': 80,
    'a': 1,
    'ds': 0.05,
    'tau': 0.15085776558260747,
    'v': 47.75363911922214,
    'dl': 0.06,
    'r': 0.05
}
sol = jump.solve(cs)
cs['dsm'] = -np.min(sol.y[1,:]) # spring travel range that needs to be matched
bounds_motion = [(-np.pi,np.pi)]+[(0.02,0.06)]*5+[(-1,1)]*1
bounds_stiffness = [(0.01,0.15)]*4+[(0.01,0.02)]+[(-1,1)]*1

xm = None
xs = None
xm = [2.6775408653380985, 0.03036417405484591, 0.0464765616668893, 0.020002243415270026, 0.0599916804667389, 0.059987624409487315, 0.2754885043894606]
xs = [0.010001175538910098, 0.024674273697069246, 0.02190461070929748, 0.04077732988661197, 0.01003214979581677, -0.14967386689132633]

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

    mass_spring = cs['mb']-0.02-np.sum(xm[1:6])*geom.tr*geom.wr*geom.rho
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
