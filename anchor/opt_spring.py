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
k = 70
a = 0.7
rot = d/r

def cb(x,convergence=0):
    print('x',x)
    print('Convergence',convergence)

def obj_stiffness(x,plot=False):
    try:
        rz,tz = stiffness.sim(x,rot,plot=plot)
    except AssertionError:
        return 10

    x_d = np.linspace(0,rot*r,50)
    f_d = -jump.f_spring(x_d,k,a,d)
    f = np.interp(x_d,rz*r,tz/r)
    e = np.sqrt(np.sum((f-f_d)**2)/f_d.shape[0])

    if plot:
        plt.figure()
        plt.plot(x_d,f_d)
        plt.plot(x_d,f,'--')

    return e

bounds_stiffness = [(0.01,0.1)]*3+[(-1,1)]*1
xs = None

# 70, 0.7, 1, 1.5
# xs = [0.01208050572905315, 0.01152624757493665, 0.012041307415044472, -0.0898142834518798]
# xs = [0.022898593635329034, 0.03982742756719852, 0.05529054326040509, -0.2624544540197491]
# xs = [0.042754394437238524, 0.05448995441932006, 0.029510363217214994, 0.4022430290121257]

# 30, 0.7, 1, 1.5
# xs = [0.02324343925017882, 0.019299122018209956, 0.01564437496558957, -0.7149529472464764]
# xs = [0.045827326070585435, 0.04303599563281178, 0.07621995188930011, -0.714119559817239]
# xs = [0.08949036347232209, 0.09996728265129545, 0.04583262770490833, 0.9706756467651072]

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

    m = (
        (np.sum(xs[:3])+geom.padf+geom.pade)*geom.tr*geom.wr-
        xs[0]*(geom.tr-stiffness.tf)*geom.wr
    )*geom.rho

    print('spring range', rot)
    print('mass',m)
    print('xs',str(list(xs)))
    print('xs error',obj_stiffness(xs,plot=True))
    plt.show()
