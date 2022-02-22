import sys
sys.path.append('../utils')

from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import numpy as np
import spin
import spin_dc
import data

I_LOAD = 0.04*0.09**2+25399/1e3/1e6
R = 10
L = 580e-6
VOLTS = [3,6,9]

def read():
    ti = np.linspace(0,1,100)
    exp = {
        't': ti
    }
    for v in VOLTS:
        dthetai = []
        for trial in [1,2]:
            # Exp motor data
            d = data.read('../data/hpcb100_{:d}V_1_{:d}.csv'.format(v,trial))
            t = np.array(d['t'])
            t1 = np.array(d['t1'])

            assert np.sum(t-t1) == 0

            x = np.array(d['x'])
            x1 = np.array(d['x1'])
            y = np.array(d['y'])
            y1 = np.array(d['y1'])
            theta = np.arctan2(y1-y,x1-x)
            dtheta = (theta[1:]-theta[:-1])/(t[1:]-t[:-1])
            t = t[1:]

            # Remove dtheta jump
            not_dtheta_jumps = dtheta < 50
            dtheta = dtheta[not_dtheta_jumps]
            t = t[not_dtheta_jumps]

            # Remove ddtehta jump
            ddtheta = np.concatenate(([0],(dtheta[1:]-dtheta[:-1])/(t[1:]-t[:-1])))
            not_ddtheta_jumps = np.abs(ddtheta) < 500
            dtheta = dtheta[not_ddtheta_jumps]
            t = t[not_ddtheta_jumps]

            # Find start index
            i = np.nonzero(t > 0.51)[0][0]
            dtheta = dtheta[i:]
            t = t[i:]
            t = t-t[0]

            # Find end index
            i = np.nonzero(t > 1)[0][0]
            dtheta = dtheta[:i]
            t = t[:i]

            # Flip direction
            dtheta = -dtheta

            dthetai.append(np.interp(ti,t,dtheta))

        dthetai = np.sum(dthetai,axis=0)/len(dthetai)
        # plt.plot(ti,dthetai)
        # plt.show()
        exp[v] = dthetai
    return exp

exp = read()

def obj(x,plot=False):
    e = 0
    sim = {}
    for v in VOLTS:
        cs = {
            'V': v,
            'J': I_LOAD,
            'b': x[0],
            'K': x[1],
            'R': R,
            'L': L
        }
        sol = spin_dc.solve(cs)
        t = sol.t
        w = sol.y[0,:]

        wi = np.interp(exp['t'],t,w)
        e += np.sqrt(np.sum((wi-exp[v])**2)/len(exp['t']))
        sim[v] = sol
    e = e/len(VOLTS)

    if plot:
        plt.figure()
        for v in VOLTS:
            plt.plot(exp['t'],exp[v])
            plt.plot(sim[v].t,sim[v].y[0,:])

        plt.figure()
        for v in VOLTS:
            plt.plot(sim[v].t,sim[v].y[1,:])
    return e

def cb(x,convergence=0):
    print('x',x)
    print('Convergence',convergence)

bounds=[(0,0.01),(0,0.2)]
x = None
x = [0.0009869882154543786, 0.13423431546352482]

if __name__ == '__main__':
    if x is None:
        res = differential_evolution(
            obj,
            bounds=bounds,
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

    print(obj(x,plot=True))
    plt.show()
