import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, NonlinearConstraint
from . import geom
from . import motion
from . import stiffness
from . import design
from template import jump

cs = jump.cs
k = 0.15
a = 1.0

def cb(x,convergence=0):
    print('x',x)
    print('Convergence',convergence)

def mass(x):
    m = (
        (np.sum(x[:4])+geom.pade)*geom.tr*geom.wr-
        x[1]*(geom.tr-stiffness.tf)*geom.wr
    )*geom.rho

    return m

def obj_stiffness(x,plot=False):
    try:
        rz,tz = stiffness.sim(x,cs['t']/k,plot=plot)
    except AssertionError:
        return 10

    tz_d = jump.t_spring(rz,k,a,cs['t'])
    e = np.sqrt(np.sum((tz-tz_d)**2)/tz.shape[0])

    if plot:
        plt.figure()
        plt.plot(rz,tz_d,'--')
        plt.plot(rz,tz,'-')

    return e+10*mass(x)

bounds_stiffness = [(0.01,0.12)]+[(0.02,0.1)]+[(0.01,0.12)]*2+[(-1,1)]*1
xs = None
# xs = [0.02098737676215514, 0.02000157482644971, 0.031158661522182295, 0.02011035646359493, -0.06936676431591338]
xs = [0.010001028703056213, 0.026324876255990545, 0.014613071145120171, 0.03307675893886909, -0.05217295522036536]
# xs = [0.010000193212392282, 0.06805170765240513, 0.06984429688212665, 0.03944228565453706, 0.5083215623745443]

# xs = [0.03276837888618742, 0.020000238768767983, 0.02906232102419351, 0.03288540681604759, -0.6873603267482058]
# xs = [0.01300840723240837, 0.020003011892568826, 0.013074232561321793, 0.03804003744669121, -0.6073977862028517]
# xs = [0.01000206120443254, 0.04116346265469098, 0.041145243337954815, 0.020846998480225094, 0.8898879289760872]

if __name__ == '__main__':
    if xs is None:
        res = differential_evolution(
            obj_stiffness,
            bounds=bounds_stiffness,
            popsize=20,
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

    print('spring range', cs['t']/k)
    print('mass',mass(xs))
    print('xs',str(list(xs)))
    print('xs error',obj_stiffness(xs,plot=True))
    plt.show()
