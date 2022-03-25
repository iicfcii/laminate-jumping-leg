import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, NonlinearConstraint
from . import geom
from . import motion
from . import stiffness
from template import jump

cs = jump.cs
d = cs['dl']
r = cs['r']

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

bounds_motion = [(-np.pi,np.pi)]+[(0.01,0.06)]*5+[(-1,1)]*1

xm = None
xs = None
xm = [2.825178219357225, 0.019652230845575398, 0.049581597690582, 0.010001644395921317, 0.059999785028978105, 0.05607371091295432, 0.10830546065402658]

if __name__ == '__main__':
    if xm is None:
        res = differential_evolution(
            obj_motion,
            bounds=bounds_motion,
            popsize=20,
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

    m = (np.sum(xm[1:6])+0.006)*geom.tr*geom.wr*geom.rho

    print('crank range', d/r)
    print('mass',m)
    print('xm',str(list(xm)))
    print('xm error',obj_motion(xm,plot=True))
    plt.show()
