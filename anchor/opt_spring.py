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

bounds_stiffness = [(0.01,0.03)]+[(0.04,0.06)]+[(0.02,0.08)]*2+[(-1,1)]+[(-1,1)]+[(0.01,0.015)]
xs = None
# k=0.3 a=0.5,1,2
# xs = [0.02186269045667044, 0.059977289524436, 0.06558116224967792, 0.028181339996950373, -0.3000530456905235, -0.774647892791124, 0.010002613245065716]
# xs = [0.023985838184202805, 0.04000000071895357, 0.020000005115256923, 0.06717759349481817, -0.8100808203036582, 0.36481598031725215, 0.014999999604143176]
# xs = [0.020788475265921157, 0.0557009449661379, 0.05472550295479147, 0.04020020432617301, 0.011441199536229796, 0.9044229959338346, 0.010001729424051913]

# k=0.15 a=0.5,1,2
# xs = [0.029999156961456424, 0.04000007076950689, 0.05259433326873095, 0.04116054754439284, -0.3193705469277317, 0.28437533111648583, 0.014999545774777905]
xs = [0.010002590722000595, 0.040003523201885265, 0.02001077865876251, 0.05518435390981338, -0.5319539744071047, 0.7684163803403559, 0.011930059441145523]
# xs = [0.010002555891985891, 0.05999630487340041, 0.0757448868158151, 0.04781397461265492, 0.522752264617517, 0.6954841332200223, 0.010000111224604945]

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
