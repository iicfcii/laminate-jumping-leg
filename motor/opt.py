import sys
sys.path.append('../utils')

from scipy.signal import butter,sosfiltfilt
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import numpy as np
import spin
import data

exp = []
for trial in [1,2]:
    # Exp motor data
    d = data.read('../data/HPCB100_9V_{:d}.csv'.format(trial),skip=6)
    t = np.array(d['Time (Seconds)'])
    x = np.array(d['X'])
    x1 = np.array(d['X1'])
    z = np.array(d['Z'])
    z1 = np.array(d['Z1'])
    theta = np.arctan2(z1-z,x1-x)
    dtheta = (theta[1:]-theta[:-1])/(t[1:]-t[:-1])
    t = t[1:]

    # Remove dtheta jump
    not_dtheta_jumps = dtheta < 50
    dtheta = dtheta[not_dtheta_jumps]
    t = t[not_dtheta_jumps]

    # Remove ddtehta jump
    ddtheta = np.concatenate(([0],(dtheta[1:]-dtheta[:-1])/(t[1:]-t[:-1])))
    not_ddtheta_jumps = np.abs(ddtheta) < 1000
    dtheta = dtheta[not_ddtheta_jumps]
    t = t[not_ddtheta_jumps]

    # Find start index
    i = np.nonzero(dtheta < -0.1)[0][0]
    dtheta = dtheta[i:]
    t = t[i:]
    t = t-t[0]

    # Find end index
    i = np.nonzero(t > 1)[0][0]
    dtheta = dtheta[:i]
    t = t[:i]

    # Flip direction
    dtheta = -dtheta

    exp.append((t,dtheta))

TAU = 1.6*9.81/100
V = 330/60*2*np.pi

def obj(x,plot=False):
    cs = {
        'tau': x[0],
        'v': x[1],
        'I': 0.04*0.09**2+25399/1e3/1e6
    }
    sol = spin.solve(cs)

    t = sol.t
    w = sol.y[1,:]

    e = 0
    for d in exp:
        we = np.interp(t,d[0],d[1])
        e += np.sqrt(np.sum((we-w)**2)/len(t))
    e = e/len(exp)

    if plot:
        plt.figure()
        for d in exp:
            we = np.interp(t,d[0],d[1])
            plt.plot(t,we)
        plt.plot(t,w)

    return e

def cb(x,convergence=0):
    print('x',x)
    print('Convergence',convergence)

if __name__ == '__main__':
    x = None
    # x = [0.09880525551823807, 33.390568656662325] # 6V
    x = [0.15085776558260747, 47.75363911922214] # 9V

    if x is None:
        res = differential_evolution(
            obj,
            bounds=[(TAU*0.5,TAU*1.5),(V*0.5,V*1.5)],
            popsize=10,
            maxiter=500,
            tol=0.01,
            callback=cb,
            workers=-1,
            polish=False,
            disp=True
        )
        print('Result', res.message)
        print('x',str(list(res.x)))
        print('Cost', res.fun)
        x = res.x

    obj(x,plot=True)
    plt.show()
