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
        (np.sum(x[:4])-x[1]+geom.pade+0.006)*(stiffness.tr+stiffness.tf(x[5]))+
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
# xs = [0.029924067038578563, 0.07439489633867016, 0.09998304774847852, 0.028418615253280186, -0.2940662786965126, -0.5614303695260598]
# xs = [0.010000221462529895, 0.09999705008137716, 0.031196740789226473, 0.0999939498970337, -0.7438160665414164, -0.21704916010943698]
# xs = [0.021341447410895064, 0.06333915626783147, 0.0606564153443763, 0.0433319842978532, 0.5312600803717236, 0.403297017832253]

# k=0.15 a=0.5,1,2
# xs = [0.016801571509922463, 0.09997271127194601, 0.09999305540532873, 0.03542469768295704, -0.4591057465364152, -0.5895869564926732]
# xs = [0.019809406068659705, 0.05000094148312782, 0.03516814960118983, 0.07516939986825065, -0.26783424601242733, 0.2235475687425712]
# xs = [0.010023357866244323, 0.08576269686428384, 0.08763175334091367, 0.04522019524055282, 0.9223652129832995, 0.19281590031139162]

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
