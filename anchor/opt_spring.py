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
a = 1

def cb(x,convergence=0):
    print('x',x)
    print('Convergence',convergence)

def mass(x):
    m = (
        (np.sum(x[:4])-x[1]+geom.pade+0.006)*(stiffness.tr+stiffness.tf(x[5]))*stiffness.wr+
        x[1]*stiffness.tf(x[5])*x[6]
    )*geom.rho

    return m

def obj_stiffness(x,plot=False):
    try:
        rz,tz = stiffness.sim(x,cs['t']/k,plot=plot)
    except AssertionError:
        return 10

    tz_d = jump.t_spring(rz,k,a,cs['t'])
    e = np.sqrt(np.sum((tz-tz_d)**2)/tz.shape[0])

    # spring should not hit ground
    ang = design.xm[0]
    l = design.xm[1:6]
    c = design.xm[6]
    ls = x[:4]
    cm = x[4]
    lk = geom.leg_spring(ang,l,c,ls,cm,motion.tr)

    yf = lk[4][1,1]
    ys = lk[6][1,1]
    if ys < yf+0.02: return 10

    if plot:
        plt.figure()
        plt.plot(rz,tz_d,'--')
        plt.plot(rz,tz,'-')

        plt.figure()
        plt.axis('scaled')
        bbox = geom.bbox(lk)
        plt.xlim(bbox[:2])
        plt.ylim(bbox[2:])
        plt.axis('scaled')
        for link in lk:
            plt.plot(link[:,0],link[:,1],'.-k')

    return e+1*mass(x)

bounds_stiffness = [(0.01,0.03)]+[(0.04,0.06)]+[(0.02,0.08)]*2+[(-1,1)]+[(0,1)]+[(0.008,0.032)]
xs = None
# k=0.3 a=0.5,1,2
# xs = [0.029998109739582446, 0.04000054050387968, 0.058885597343209924, 0.024213466259905357, -0.6114900907861596, 0.5615574543067529, 0.03199785777826078]
# xs = [0.01000293444514275, 0.04000206204808693, 0.020001874561329687, 0.05808113629218124, -0.33833302727481795, 0.29601242325089855, 0.022929041049290857]
# xs = [0.014401602291057556, 0.04098726504490992, 0.040669552143183435, 0.028935824774060828, 0.3478344773053903, 0.9443943849494187, 0.008000125135292755]

# k=0.15 a=0.5,1,2
# xs = [0.028351936374588847, 0.040000879836479854, 0.056946005315651284, 0.02276333038136599, -0.6788923995147724, 0.9383280857354669, 0.024240188197119667]
xs = [0.010004710271333525, 0.040004147809067, 0.020012070558045505, 0.0551528015298821, -0.6537171421762743, 0.8243483475852564, 0.011936762866906514]
# xs = [0.010000048924065407, 0.05999736285976756, 0.07279938028019925, 0.03967831456767372, 0.6850015510099857, 0.4237929381864047, 0.008000569324246799]

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
