import sys
sys.path.append('../template')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, LinearConstraint
import fourbar
import jump

pi = np.pi
cs = {
    'g': 9.81,
    'mb': 0.05,
    'ml': 0.001,
    'k': 100,
    'a': 1,
    'el': 0.1, # max leg extension
    'tau': 0.215,
    'v': 383/60*2*pi,
    'em': 0.1,
    'r': 0.06
}

m = 0.04
ang_limit = (-pi,pi)
l_limit = (0.04,0.12)
w_limit = (0.01,0.03)
fxb_limit = 1.5
e_max = 10

bounds = [ang_limit]+[l_limit]*5+[w_limit]*2+[(-1,1)]*1

# Desired force
x0 = [0,0,0,0]
sol = jump.solve(x0, cs)
td = sol.t
ybd = sol.y[0,:]
dybd = sol.y[2,:]

def fromX(x):
    ang = x[0]
    l = x[1:6]
    w = x[6:8]
    c = x[8]

    return ang,l,w,c

def error_yb(data):
    num_step = 100
    # Compare on the longer time range
    tdd = np.linspace(0,np.maximum(data['t'][-1],td[-1]),num_step)

    ybi = np.interp(tdd,data['t'],data['yb'])
    ybdi = np.interp(tdd,td,ybd)

    dybi = np.interp(tdd,data['t'],data['dyb'])
    dybdi = np.interp(tdd,td,dybd)

    # plt.figure()
    # plt.plot(tdd,ybdi)
    # plt.plot(tdd,ybi)
    # plt.show()

    # Normalized wrt max desired value
    e = np.sqrt(np.sum((ybi-ybdi)**2)/num_step)/np.amax(np.abs(ybdi))
    de = np.sqrt(np.sum((dybi-dybdi)**2)/num_step)/np.amax(np.abs(dybdi))
    return 0.5*e+0.5*de

def error_dyb(data):
    e = -data['dyb'][-1]
    return e

def fxb_max(data):
    ts = 1e-3
    if data['t'][-1] <= ts: return None
    return np.amax(np.abs(data['fxb'])[np.array(data['t']) > ts])

def obj(x,e):
    ang,l,w,c = fromX(x)

    try:
        data = fourbar.solve(ang,l,w,c,m,cs,vis=False)
    except AssertionError:
        return e_max

    f = fxb_max(data)
    if f is None or f > fxb_limit:
        return e_max

    return e(data)

def cb(x,convergence=0):
    ang,l,w,c = fromX(x)
    print('ang =',ang)
    print('l =',str(list(l)))
    print('w =',str(list(w)))
    print('c =',c)
    print('convergence =',convergence)

if __name__ == '__main__':
    res = differential_evolution(
        obj,
        bounds=bounds,
        args=(error_yb,),
        popsize=10,
        maxiter=1000,
        tol=0.01,
        callback=cb,
        workers=-1,
        polish=False,
        disp=True
    )
    ang,l,w,c = fromX(list(res.x))
    print('Result', res.message)
    print('ang =',ang)
    print('l =',str(list(l)))
    print('w =',str(list(w)))
    print('c =',c)
    print('Cost', res.fun)
