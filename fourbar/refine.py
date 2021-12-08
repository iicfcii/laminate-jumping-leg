import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, LinearConstraint
import opt

pi = np.pi

ang = -2.452017316719783
l = [0.037902276812561836, 0.09012769459734879, 0.04696167467809782, 0.07516468292471243, 0.09482336961226434]
kl = [0.33313110933656503, 0.36487632086031957, 0.4903842009988259, 0.1469028123266444]
c = -0.6047912443960423
dir = -0.4417283908542817
gnd = 0.31390830905413214

x0 = [ang]+l+kl
ang_r = 1
l_r = 0.05
kl_r = 0.5
bounds_ang = [(
    ang-ang_r if ang-ang_r > opt.ang_limit[0] else opt.ang_limit[0],
    ang+ang_r if ang+ang_r < opt.ang_limit[1] else opt.ang_limit[1]
)]
bounds_l = [(
    val-l_r if val-l_r > opt.l_limit[0] else opt.l_limit[0],
    val+l_r if val+l_r < opt.l_limit[1] else opt.l_limit[1]
) for val in l]
bounds_kl = [(
    val-kl_r if val-kl_r > opt.kl_limit[0] else opt.kl_limit[0],
    val+kl_r if val+kl_r < opt.kl_limit[1] else opt.kl_limit[1]
) for val in kl]
bounds = bounds_ang+bounds_l+bounds_kl
cons = [
    LinearConstraint(np.array([[0,1,1,1,1,1,0,0,0,0]]),*opt.size_limit),
]

def fromX(x):
    ang = x[0]
    l = x[1:6]
    kl = x[6:10]

    return ang,l,kl

def obj(x,e):
    ang,l,kl = fromX(x)

    try:
        data = opt.solve(ang,l,kl,c,dir,gnd,opt.cs,vis=False)
    except AssertionError:
        return 100000

    f = opt.fxb_max(data)
    if f is None or f > 10:
        return 100000

    return e(data)

def cb(x,convergence=0):
    ang,l,kl = fromX(x)
    print('ang =',ang)
    print('l =',str(list(l)))
    print('kl =',str(list(kl)))
    print('c =',c)
    print('dir =',dir)
    print('gnd =',gnd)
    print('convergence =',convergence)

if __name__ == '__main__':
    res = differential_evolution(
        obj,
        bounds=bounds,
        args=(opt.error_yb,),
        constraints=cons,
        popsize=10,
        maxiter=500,
        tol=0.01,
        callback=cb,
        workers=-1,
        polish=False,
        disp=True
    )
    ang,l,kl = fromX(list(res.x))
    print('Result', res.message)
    print('ang =',ang)
    print('l =',str(list(l)))
    print('kl =',str(list(kl)))
    print('c =',c)
    print('dir =',dir)
    print('gnd =',gnd)
    print('Cost', res.fun)
