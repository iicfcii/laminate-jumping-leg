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
a = 1.5
rot = d/r

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

    return e+10*mass(x)

bounds_stiffness = [(0.01,0.12)]+[(0.02,0.1)]+[(0.01,0.12)]*2+[(-1,1)]*1
xs = None

# 70, 0.7, 1, 1.5
# xs = [0.026603827353495563, 0.020001528458561783, 0.04210425314870689, 0.036543992076058116, -0.030767022628516272]
# xs = [0.010000045601262782, 0.02003606920216318, 0.05208638171400896, 0.06244259176458898, -0.07677562269701532]
xs = [0.01000205230570337, 0.04358801661603767, 0.06265785645661426, 0.040406023929044625, 0.8684900989979001]

# 30, 0.7, 1, 1.5
# xs = [0.010000889558147465, 0.020533479423717457, 0.01771223791479492, 0.019252796225139722, -0.6958259115762272]
# xs = [0.010001033968105644, 0.04440100825662386, 0.026927632892689006, 0.06169465103699601, -0.6348330389125217]
# xs = [0.010000641816559491, 0.09443259496277022, 0.11099903069844592, 0.05747566624650243, 0.3979551667527627]

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

    print('spring range', rot)
    print('mass',mass(xs))
    print('xs',str(list(xs)))
    print('xs error',obj_stiffness(xs,plot=True))
    plt.show()
