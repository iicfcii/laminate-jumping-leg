import sys
sys.path.append('../template')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, LinearConstraint
from fourbar import solve
import jump

pi = np.pi

bounds = [(-pi,pi)]+[(0.02,0.1)]*5+[(0.05,0.5)]*4+[(-1,1)]*3
cons = [
    LinearConstraint(np.array([[0,1,1,1,1,1,0,0,0,0,0,0,0]]),0,0.25),
]
cs = {
    'g': 9.81,
    'mb': 0.03,
    'ml': 0.01,
    'k': 100,
    'a': 1,
    'el': 0.1, # max leg extension
    'tau': 0.215,
    'v': 383/60*2*pi,
    'em': pi,
    'r': 0.04
}

# Desired force
x0 = [0,0,0,0]
sol = jump.solve(x0, cs)
td = sol.t
ybd = sol.y[0,:]

def fromX(x):
    ang = x[0]
    l = x[1:6]
    kl = x[6:10]
    c = x[10]
    dir = x[11]
    gnd = x[12]

    return ang,l,kl,c,dir,gnd

def error_yb(data):
    # Compare on the longer time range
    tdd = np.linspace(0,np.maximum(data['t'][-1],td[-1]),100)
    ybi = np.interp(tdd,data['t'],data['yb'])
    ybdi = np.interp(tdd,td,ybd)

    # plt.figure()
    # plt.plot(tdd,ybdi)
    # plt.plot(tdd,ybi)
    # plt.show()

    e = np.sqrt(np.sum((ybi-ybdi)**2))
    return e

def error_dyb(data):
    e = -data['dyb'][-1]
    return e

def fxb_max(data):
    ts = 1e-3
    if data['t'][-1] <= ts: return None
    return np.amax(np.abs(data['fxb'])[np.array(data['t']) > ts])

def obj(x,e):
    ang,l,kl,c,dir,gnd = fromX(x)

    try:
        data = solve(ang,l,kl,c,dir,gnd,cs,vis=False)
    except AssertionError:
        return 100000

    f = fxb_max(data)
    if f is None or f > 1:
        return 100000

    return e(data)

def cb(x,convergence=0):
    ang,l,kl,c,dir,gnd = fromX(x)
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
        args=(error_yb,),
        constraints=cons,
        popsize=10,
        maxiter=500,
        tol=0.1,
        callback=cb,
        workers=-1,
        polish=False,
        disp=True
    )
    ang,l,kl,c,dir,gnd = fromX(list(res.x))
    print('Result', res.message)
    print('ang =',ang)
    print('l =',str(list(l)))
    print('kl =',str(list(kl)))
    print('c =',c)
    print('dir =',dir)
    print('gnd =',gnd)
    print('Cost', res.fun)
