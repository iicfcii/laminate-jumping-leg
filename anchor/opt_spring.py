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
        (np.sum(x[:4])-x[1]+geom.pade)*geom.tr*geom.wr+
        x[1]*stiffness.tf*x[5]
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

    return e+1*mass(x)

bounds_stiffness = [(0.01,0.1)]+[(0.05,0.1)]+[(0.01,0.1)]*2+[(-1,1)]+[(0.01,0.04)]
xs = None
# k=0.15 a=0.5,1,2
# xs = [0.03978171855634041, 0.050002477119788585, 0.07510506897492346, 0.02105574222283238, -0.5996360860805907, 0.03996431775016607]
# xs = [0.010004703307973518, 0.05000083065349166, 0.02311882052712757, 0.06753132964751529, -0.8332206542017118, 0.015471341702232673]
# xs = [0.01000165934550968, 0.08959909694271116, 0.08987598479118736, 0.036645855203370964, 0.35657535377504856, 0.010001099186796161]

# k=0.3 a=0.5,1,2
# xs = [0.065931945802239, 0.05000054789373298, 0.08823412496044417, 0.034582652881715165, -0.4283124779052667, 0.039963375849935354]
# xs = [0.010187354560972355, 0.05000413047456115, 0.02387997909166409, 0.06968820992406445, -0.9351342744057919, 0.030291249656529164]
# xs = [0.010002543185644142, 0.050022795862418486, 0.04983374238454475, 0.020484050111783274, 0.1298032531772193, 0.010334691539600675]

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
