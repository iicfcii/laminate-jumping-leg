from scipy.optimize import minimize,differential_evolution
import matplotlib.pyplot as plt
import numpy as np
from utils import data
from template import jump
from . import sim

def center(plot=False):
    def obj(x):
        xc,yc,r = x
        e = (xm-xc)**2+(ym-yc)**2-r**2
        return np.sqrt(np.mean(e**2))

    xs = []
    for i,k in enumerate([0.1,0.2]):
        for a in [2]:
            for n in [1,2,3]:
                d = data.read('./data/damping/damping_{:d}_{:d}_{:d}_{:d}.csv'.format(1,int(k*100),int(a*10),n))
                t = np.array(d['t'])
                xm = np.array(d['x'])
                ym = np.array(d['y'])

                x0 = [xm[0],ym[0],0.018]
                res = minimize(obj,x0,tol=1e-6)
                xc,yc,r = res.x
                xs.append([xc,yc,np.abs(r)])

                if plot:
                    plt.figure('center')
                    plt.plot(xm,ym,'.',color='C{:d}'.format(i))

    xs = np.array(xs)
    xc,yc,r = np.mean(xs,axis=0)

    if plot:
        thetap = np.linspace(-np.pi,0,100)
        xp = r*np.cos(thetap)+xc
        yp = r*np.sin(thetap)+yc
        print(xp)
        plt.plot(xp,yp,'k')
        plt.axis('scaled')

    return xc,yc,r

def read(k,a,n,c,plot=False):
    xc,yc,r = c

    # Read data
    d = data.read('./data/damping/damping_{:d}_{:d}_{:d}_{:d}.csv'.format(1,int(k*100),int(a*10),n))
    t = np.array(d['t'])
    xm = np.array(d['x'])
    ym = np.array(d['y'])

    # Convert to angle
    theta = np.arctan2(ym-yc,xm-xc)
    theta0 = np.mean(theta[t > 3])
    theta -= theta0

    # Select range
    dtheta = np.concatenate([[0],(theta[1:]-theta[:-1])/(t[1:]-t[:-1])])
    i = np.nonzero(dtheta<-10)[0][0]-1
    j = np.nonzero(t > t[i]+sim.tfinal)[0][0]
    tp = t[i:j]
    thetap = theta[i:j]
    dthetap = dtheta[i:j]

    if plot:
        plt.figure('theta')
        plt.plot(t,theta)
        plt.plot(tp,thetap)

    tp -= tp[0]
    return tp,thetap

def obj(x,d,plot=False):
    k,b = x
    e = 0

    for i, (t_e,theta_e) in enumerate(d):
        cs = {
            'k': k,
            'b': b,
            'I': sim.cs['I'],
            'theta0': theta_e[0]
        }
        sol = sim.solve(cs)
        t = sol.t
        theta = sol.y[0,:]
        theta_e = np.interp(t,t_e,theta_e)

        e += np.sqrt(np.mean((theta-theta_e)**2))

        if plot:
            plt.figure('match')
            plt.subplot(3,1,i+1)
            plt.plot(t,theta)
            plt.plot(t,theta_e)

    return e

def cb(x,convergence=0):
    print('x',x)
    print('Convergence',convergence)

k = 0.2
a = 2
bounds=[(0,0.3),(0,0.001)]
x = None

# k=0.1 a=0.5,1,2
# x = [0.036696747822411474, 2.9593453947478416e-05]

# k=0.2 a=0.5,1,2
x = [0.06043227741613831, 7.067560971637805e-05]

if __name__ == '__main__':
    c = center(plot=False)
    d = []
    for n in [1,2,3]:
        t_e, theta_e = read(k,a,n,c)
        d.append([t_e, theta_e])

    if x is None:
        res = differential_evolution(
            obj,
            bounds=bounds,
            args=(d,),
            popsize=10,
            maxiter=500,
            tol=0.001,
            callback=cb,
            workers=-1,
            polish=False,
            disp=True
        )
        print('Result', res.message)
        print('x',str(list(res.x)))
        print('Cost', res.fun)
        x = res.x

    print(obj(x,d,plot=True))
    plt.show()
