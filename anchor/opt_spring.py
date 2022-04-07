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

bounds_stiffness = [(0.01,0.03)]+[(0.04,0.06)]+[(0.02,0.08)]*2+[(-1,1)]+[(-1,1)]+[(0.008,0.012)]
xs = None
# k=0.3 a=0.5,1,2
# xs = [0.02714941282400844, 0.058318428084070374, 0.06726707044028961, 0.030050558112119347, -0.8449617610949873, -0.8775245556709352, 0.008000407428149852]
# xs = [0.027743385626954022, 0.04000085781603069, 0.021556963206502664, 0.07002472310991947, -0.9559603674149335, 0.2657320561271106, 0.01199991818090493]
# xs = [0.020835778101677412, 0.05283309916850176, 0.05254727219341254, 0.03931031086950555, 0.7432129742078131, 0.9770970464948663, 0.008000746522243668]

# k=0.15 a=0.5,1,2
# xs = [0.029998066126279796, 0.040001598826999406, 0.05159225649981301, 0.046203276583518894, -0.16808220012570174, 0.3499385755941311, 0.01199968502850087]
xs = [0.010000071967791437, 0.04000013664872082, 0.020007145194154346, 0.05517785183637364, -0.09454095853254785, 0.5711112162846819, 0.010497117659487822]
# xs = [0.01000079679076422, 0.05999712177996903, 0.07958562611938018, 0.04627144167627453, 0.618119469185275, 0.4763325933953617, 0.008000260877909948]

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
