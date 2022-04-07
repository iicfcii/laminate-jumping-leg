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
# xs = [0.027017593073951657, 0.05252219442844985, 0.06448638167576892, 0.02702476316030279, -0.3382138742160071, -0.05260006621906821, 0.008000680883143686]
# xs = [0.028606878978159302, 0.040000099038528386, 0.050072059636600164, 0.07999895986169758, 0.16871148505353584, 0.24459833005514353, 0.01199881721535646]
# xs = [0.01985795777359114, 0.048005697729447905, 0.04848550258818857, 0.03673065929983377, 0.17004094378072154, 0.0758745410612276, 0.008000499230231236]

# k=0.15 a=0.5,1,2
# xs = [0.029999913367828868, 0.04000218793697699, 0.05109938292140102, 0.05336172624075251, -0.5745215940011114, 0.9593606506785057, 0.011999459838686919]
xs = [0.010001411740461421, 0.04000018476745831, 0.02001651339793885, 0.05516645498619387, -0.20383837688220796, 0.7025844850711671, 0.011931051557821181]
# xs = [0.010000283253409084, 0.059997398141954164, 0.07431233167652107, 0.040495522954785514, 0.41514508771138003, 0.34053085853774445, 0.008000507654752905]

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
