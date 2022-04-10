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
# xs = [0.02999985514404422, 0.04000147235108945, 0.0580829250987848, 0.025881459725850188, -0.49268869288100847, 0.849313912807092, 0.04499949096753275]
# xs = [0.010003403256441088, 0.04000283663114815, 0.020002648651013977, 0.05810654518124439, -0.25791933642059417, 0.06556389159398113, 0.033619967488176444]
# xs = [0.018392848881951782, 0.042293860183318856, 0.0434966910332138, 0.03338184353612353, 0.4916003410457195, 0.17437952849571753, 0.010003648047905513]

# k=0.15 a=0.5,1,2
# xs = [0.029996539939840708, 0.04000029657986639, 0.05777319153293769, 0.023548932817089564, -0.05785759709519178, 0.4672141853774169, 0.03474599316106239]
xs = [0.010003862581003838, 0.04000029556238622, 0.020073597654380805, 0.05521744651629045, -0.4433834529905818, 0.7499975016973623, 0.017491496486074527]
# xs = [0.01000124215875559, 0.04999909767449563, 0.07427630058663459, 0.04417220191867211, 0.27191500014323544, 0.137397005521798, 0.01000000705641909]

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
