import sys
sys.path.append('../template')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, NonlinearConstraint
import fourbar
import jump

pi = np.pi
cs = {
    'g': 9.81,
    'mb': 0.03,
    'ml': 0.0001,
    'k': 30,
    'a': 0.3,
    'ds': 0.05,
    'tau': 0.109872,
    'v': 30.410616886749196,
    'dl': 0.06,
    'r': 0.05
}

m = 0.02
ang_limit = (-pi,pi)
l_limit = (0.02,0.1)
w_limit = (0.01,0.03)
fxb_limit = 10
e_max = 10

def total_mass(l,w):
    ml = (fourbar.tr*fourbar.wr*(l[0]+l[2]+l[3])+fourbar.tf*w[0]*(l[1])+fourbar.tf*w[1]*(l[4]))*fourbar.rho
    return m + ml

def mass_con(x):
    ang,l,w,c = fromX(x)
    return total_mass(l,w)

eps = 5e-4
bounds = [ang_limit]+[l_limit]*5+[w_limit]*2+[(-1,1)]*1
cons = [
    NonlinearConstraint(mass_con,cs['mb']-eps,cs['mb']+eps)
]

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
        constraints=cons,
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
