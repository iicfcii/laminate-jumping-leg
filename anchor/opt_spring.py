import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, NonlinearConstraint
from . import geom
from . import motion
from . import stiffness
from . import design
from template import jump

k = 0.2
a = 1
cs = jump.cs
cs['k'] = k
cs['a'] = a

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

    rzp = np.zeros((6,rz.shape[0]))
    rzp[4,:] = rz
    tz_d = jump.f_ts(rzp,cs)
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

bounds_stiffness = [(0.01,0.03)]+[(0.04,0.05)]+[(0.02,0.08)]*2+[(-1,1)]+[(-1,1)]+[(0.01,0.04)]
xs = None
# k=0.2 a=0.5,1,2
# xs = [0.02999086184546148, 0.040002107461506745, 0.06250412764681808, 0.02543020424596264, -0.6359082134149612, 0.50311364133524, 0.031134344485818217]
xs = [0.010005448787907288, 0.04000426750334961, 0.020008446255142802, 0.06426816909208403, -0.907274422962836, 0.04631181879835111, 0.024278750725350746]
# xs = [0.02999250757073879, 0.040001779112095986, 0.04882313390468029, 0.043871504742187276, 0.19868998028173146, -0.20678056846447535, 0.02284021640551265]

# k=0.1 a=0.5,1,2
# xs = [0.029978987894644666, 0.040004736484183075, 0.06251082132970592, 0.025424985719201723, -0.47321130136626266, 0.08661482489275896, 0.021278422489050344]
# xs = [0.010001146968225004, 0.04000076364858595, 0.020000225850462402, 0.06312479209574236, -0.23537948854795043, 0.345418529184057, 0.01226921603916778]
# xs = [0.020084156047150498, 0.04000506898759209, 0.053548831756446987, 0.039550728715467706, 0.7425352329340649, -0.09467439398151001, 0.013794232785892391]

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
