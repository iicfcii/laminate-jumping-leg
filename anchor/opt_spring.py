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
a = 0.5

def cb(x,convergence=0):
    print('x',x)
    print('Convergence',convergence)

def mass(x):
    m = (
        (np.sum(x[:4])-x[1]+geom.pade)*(stiffness.tr+stiffness.tf(x[5]))+
        x[1]*stiffness.tf(x[5])
    )*geom.rho*stiffness.wr

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

    return e+1*mass(x)

bounds_stiffness = [(0.01,0.03)]+[(0.05,0.1)]+[(0.01,0.1)]*2+[(-1,1)]+[(-1,1)]
xs = None
# k=0.3 a=0.5,1,2
# xs = [0.02786552989186319, 0.06419191700680094, 0.08738387878548401, 0.02544192931562921, -0.36480065026094866, -0.5598520845253099]
# xs = [0.010000065502478067, 0.08613905390334783, 0.026124476243007223, 0.09999046218291263, -0.6843952260273416, -0.2166092391236507]
# xs = [0.020446446804854225, 0.05279617296324947, 0.05045967897283137, 0.038160590940863964, 0.6482294758287703, 0.04220408871679293]

# k=0.15 a=0.5,1,2
# xs = [0.02028976334849916, 0.08556772818298769, 0.0999991625395486, 0.02908366145491546, -0.6347665286412476, -0.4780881516601482]
# xs = [0.02894489301271689, 0.05000258370067141, 0.03659364563040861, 0.08016719872632525, -0.6636194299269107, 0.16558504999740875]
# xs = [0.01013319185250802, 0.0721549852339859, 0.07048035104982785, 0.03786895477655124, 0.2032629803301509, 0.5773366791295995]

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
