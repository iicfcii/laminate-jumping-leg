import sys
sys.path.append('../utils')

from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import numpy as np
import spin
import spin_dc
import data

TAU = 1.6*9.81/100
V = 330/60*2*np.pi
I_LOAD = 0.04*0.09**2+25399/1e3/1e6
K = 1/0.1*9.81/1000
VOLTS = [6,9]

DC = True

def read():
    ti = np.linspace(0,1,100)
    exp = {
        't': ti
    }
    for v in VOLTS:
        dthetai = []
        for trial in [1,2]:
            # Exp motor data
            d = data.read('../data/HPCB100_{:d}V_{:d}.csv'.format(v,trial),skip=6)
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

            dthetai.append(np.interp(ti,t,dtheta))

        dthetai = np.sum(dthetai,axis=0)/len(dthetai)
        exp[v] = dthetai
    return exp

exp = read()

def obj(x,plot=False):
    e = 0
    sim = {
        't': exp['t']
    }
    for v in VOLTS:
        if not DC:
            r = v/9
            cs = {
                'tau': x[0]*r,
                'v': x[1]*r,
                'I': I_LOAD
            }
            sol = spin.solve(cs)
            t = sol.t
            w = sol.y[1,:]
        else:
            r = v/9
            cs = {
                'V': v,
                'J': I_LOAD,
                'K': 1/0.1*9.81/1000,
                'b': x[0],
                'R': x[1],
                'L': x[2]
            }
            sol = spin_dc.solve(cs)
            t = sol.t
            w = sol.y[0,:]

        wi = np.interp(exp['t'],t,w)
        e += np.sqrt(np.sum((wi-exp[v])**2)/len(exp['t']))
        sim[v] = wi
    e = e/len(VOLTS)

    if plot:
        plt.figure()
        for v in VOLTS:
            plt.plot(exp['t'],exp[v])
            plt.plot(sim['t'],sim[v])

    return e

def cb(x,convergence=0):
    print('x',x)
    print('Convergence',convergence)

if not DC:
    bounds=[(TAU*0.5,TAU*1.5),(V*0.5,V*1.5)]
else:
    bounds=[(0,0.1),(0,1),(0,1)]

if __name__ == '__main__':
    x = None
    if not DC:
        x = [0.15085776558260747, 47.75363911922214]
    else:
        x = [0.0609658891195507, 0.14311864123002638, 0.03306177446894332]

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
