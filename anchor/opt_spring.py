import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, NonlinearConstraint
from . import geom
from . import motion
from . import stiffness
from . import design
from template import jump

cs = jump.cs
d = cs['dl']
r = cs['r']
k = 70
a = 1.5

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
        rz,tz = stiffness.sim(x,d/r,plot=plot)
    except AssertionError:
        return 10

    rots = -rz+design.xm[0]
    y = motion.sim(rots,design.xm,plot=False)[1]
    y = y[0]-y
    f = tz/motion.simf(rots,design.xm,plot=False)
    f_d = -jump.f_spring(y,k,a,d)
    e = np.sqrt(np.sum((f-f_d)**2)/f.shape[0])

    if plot:
        plt.figure()
        plt.plot(y,f_d)
        plt.plot(y,f,'--')

    return e+10*mass(x)

bounds_stiffness = [(0.01,0.12)]+[(0.02,0.1)]+[(0.01,0.12)]*2+[(-1,1)]*1
xs = None

# 70, 0.7, 1, 1.5
# xs = [0.02453676694574331, 0.020000732385687246, 0.10125899577516081, 0.08413416219089748, -0.1265235226319681]
# xs = [0.013133321760049825, 0.020000078368385035, 0.06272316290185735, 0.06296665182395636, -0.1137145307606875]
xs = [0.010000789178548651, 0.036150833359109166, 0.06154262069317969, 0.048362471842380766, 0.8984928639951657]

# 30, 0.7, 1, 1.5
# xs = [0.010000160448219449, 0.02181965569618964, 0.04563078349701109, 0.024644261061567842, -0.278723712733692]
# xs = [0.010001717564638218, 0.03653465111203534, 0.03256468844284267, 0.040882653553750786, -0.21319691440832855]
# xs = [0.010010504712925354, 0.07836488640313623, 0.10290343632345197, 0.06998081224734834, 0.0987970095253875]

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

    print('spring range', d/r)
    print('mass',mass(xs))
    print('xs',str(list(xs)))
    print('xs error',obj_stiffness(xs,plot=True))
    plt.show()
