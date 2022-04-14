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

bounds_stiffness = [(0.01,0.03)]+[(0.04,0.05)]+[(0.02,0.08)]*2+[(-1,1)]+[(0,1)]+[(0.01,0.045)]
xs = None
# k=0.3 a=0.5,1,2
# xs = [0.029998704676231024, 0.04000179079101605, 0.058869077553301216, 0.03239256417557532, -0.4754310894724434, 0.7511246313685815, 0.04499890588146831]
# xs = [0.010011472619069256, 0.04000417061941586, 0.020001656493324835, 0.06412132159984196, -0.2770677643787812, 0.21740048840448367, 0.03644779912524001]
# xs = [0.02184589428274585, 0.041999944116019615, 0.049685220233965675, 0.0380291482571013, 0.8687947888984788, 0.16631876927315137, 0.010000638035533471]

# k=0.15 a=0.5,1,2
# xs = [0.029999727694895256, 0.04000139717905318, 0.06275517682487114, 0.025290006887478636, -0.31358933284138524, 0.69163123870495, 0.036864976542153406]
xs = [0.010000059573823779, 0.04000600476557665, 0.020018061269938433, 0.06154937174042545, -0.8160034077466766, 0.8583890432968987, 0.018793583766936156]
# xs = [0.010000173930225925, 0.049997980795558145, 0.06932107827100856, 0.037047049180828316, 0.809863502784785, 0.5853528355725459, 0.010000062830065694]

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
