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

bounds_stiffness = [(0.01,0.03)]+[(0.05,0.06)]+[(0.02,0.08)]*2+[(-1,1)]+[(-1,1)]+[(0.01,0.02)]
xs = None
# k=0.3 a=0.5,1,2
# xs = [0.021889214208886485, 0.05999649123722096, 0.06560257835934695, 0.02820276183740586, -0.8917708930362038, -0.6419219270495966, 0.01000292887135154]
# xs = [0.026434542875525994, 0.050000761204795435, 0.02097958038853082, 0.07930216927615023, -0.6191628447244071, 0.6513003941431224, 0.01999886433502342]
# xs = [0.020921775866681945, 0.055860505377171354, 0.05486143001722995, 0.04039649847068741, 0.836297190020123, 0.31643474673850536, 0.01000082664926574]

# k=0.15 a=0.5,1,2
# xs = [0.029996084886386637, 0.050001956242265985, 0.05457468394922662, 0.04856663062286302, -0.30175664614785436, 0.738812191821647, 0.019999717186899104]
xs = [0.010000393196209191, 0.050001221090524975, 0.02001164401615783, 0.06450444217669012, -0.2947252003143557, 0.8904576489128688, 0.015855483520656254]
# xs = [0.010000295581336865, 0.059998297003890554, 0.07508763115297416, 0.047646130212618786, 0.4895857028987444, 0.4605334803477552, 0.01000028543318548]

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
