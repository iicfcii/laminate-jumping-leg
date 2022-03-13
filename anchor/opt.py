import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, NonlinearConstraint
from . import geom
from . import motion
from . import stiffness
from template import jump

cs = jump.cs
cs['k'] = 70
cs['a'] = 1.0
cs['r'] = 0.06

d = cs['dl']
r = cs['r']
k = cs['k']
a = cs['a']

sol = jump.solve(cs,plot=False)
rot = -np.amin(sol.y[1,:])/r

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
        (x[0]+x[2]+x[3]+geom.pad)*geom.tr*geom.wr*geom.rho+
        (x[1]-geom.pad)*geom.tf*x[4]*geom.rho
    )
    em = np.abs(m-mass_spring)

    if plot:
        plt.figure()
        plt.plot(x_d,f_d)
        plt.plot(x_d,f,'--')
        print('mass error',em)

    return e+5*em

bounds_motion = [(-np.pi,np.pi)]+[(0.01,0.06)]*5+[(-1,1)]*1
bounds_stiffness = [(0.01,0.1)]+[(0.02+geom.pad,0.1+geom.pad)]+[(0.01,0.1)]*2+[(0.01,0.015)]+[(0,1)]*1

xm = None
xs = None
xm = [2.829275067178406, 0.01963287664998194, 0.04961462878068207, 0.010005344360866891, 0.0599943360502974, 0.056280347200330344, 0.6257648591555063]
xs = [0.010000847020704577, 0.024450100750967, 0.015202591784812133, 0.03745824779270103, 0.010001328303363446, -0.39481221824562496]

if __name__ == '__main__':
    if xm is None:
        res = differential_evolution(
            obj_motion,
            bounds=bounds_motion,
            popsize=50,
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

    mass_fourbar = np.sum(xm[1:6])*geom.tr*geom.wr*geom.rho
    mass_spring = 0.007-mass_fourbar
    assert mass_spring > 0.001

    if xs is None:
        res = differential_evolution(
            obj_stiffness,
            bounds=bounds_stiffness,
            args=(mass_spring,),
            popsize=10,
            maxiter=500,
            tol=0.001,
            callback=cb,
            workers=-1,
            polish=False,
            disp=True
        )
        print('Result', res.message)
        print('xs',str(list(res.x)))
        print('Cost', res.fun)
        xs = res.x

    print('crank range', d/r)
    print('spring Mass', mass_spring)
    print('spring range', rot)
    print('xm',str(list(xm)))
    print('xs',str(list(xs)))
    print('xm error',obj_motion(xm,plot=True))
    print('xs error',obj_stiffness(xs,mass_spring,plot=True))
    plt.show()
