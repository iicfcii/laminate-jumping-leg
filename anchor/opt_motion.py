import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, NonlinearConstraint
from . import geom
from . import motion
from . import stiffness
from template import jump

cs = jump.cs

def mass(x):
    return (np.sum(x[1:6])+0.006)*motion.tr*motion.wr*geom.rho

def obj_motion(x,plot=False):
    rots = np.linspace(0,-cs['d'],50)+x[0]
    try:
        xs,ys = motion.sim(rots,x,plot=plot)
    except AssertionError:
        return 1

    xs_d = np.zeros(xs.shape)
    ys_d = (rots-x[0])*cs['r']+ys[0]

    ex = np.sqrt(np.sum((xs-xs_d)**2)/xs.shape[0])
    ey = np.sqrt(np.sum((ys-ys_d)**2)/ys.shape[0])

    if mass(x) > 0.005: return 1

    if plot:
        plt.figure()
        plt.plot(rots,ys_d)
        plt.plot(rots,ys,'.')
        plt.plot(rots,xs_d)
        plt.plot(rots,xs,'.')

    return ex+ey

def cb(x,convergence=0):
    print('x',x)
    print('Convergence',convergence)

bounds_motion = [(-np.pi,np.pi)]+[(0.01,0.07)]*5+[(-1,1)]*1

xm = None
xs = None
xm = [2.871686052424245, 0.03122667210051749, 0.03952791136488216, 0.02074981796437542, 0.05310881494281977, 0.05130490605022549, 0.2217093190577344]

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


    print('crank range', cs['d'])
    print('mass',mass(xm))
    print('xm',str(list(xm)))
    print('xm error',obj_motion(xm,plot=True))

    rots = np.linspace(0,-cs['d'],50)+xm[0]
    rs = motion.simf(rots,xm,plot=False)
    rots = xm[0]-rots
    p = np.polyfit(rots,rs,2)
    rsp = np.polyval(p,rots)

    plt.figure()
    plt.plot(rots,rs)
    plt.plot(rots,rsp)
    print('r coeff',str(list(p)))

    plt.show()
