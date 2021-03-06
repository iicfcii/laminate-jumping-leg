from scipy.optimize import minimize,differential_evolution
import matplotlib.pyplot as plt
import numpy as np
from utils import data
from template import jump
from . import sim

def center(plot=False):
    def obj(x):
        xc,yc = x
        rs = np.sqrt((xm-xc)**2+(ym-yc)**2)
        return np.std(rs)

    xm = []
    ym = []
    for i,k in enumerate([0.1,0.2]):
        for a in [0.5,1,2]:
            for n in [1,2,3]:
                d = data.read('./data/damping/damping_{:d}_{:d}_{:d}_{:d}.csv'.format(1,int(k*100),int(a*10),n))
                xm += d['x']
                ym += d['y']
    xm = np.array(xm)
    ym = np.array(ym)

    x0 = [xm[0],ym[0]]
    res = minimize(obj,x0,tol=1e-6)
    print(res)
    xc,yc = res.x

    if plot:
        r = np.mean(np.sqrt((xm-xc)**2+(ym-yc)**2))
        thetap = np.linspace(-np.pi,np.pi,100)
        xp = r*np.cos(thetap)+xc
        yp = r*np.sin(thetap)+yc
        plt.plot(xp,yp)
        plt.plot(xm,ym,'.')
        plt.axis('scaled')

    return xc,yc

def read(k,a,n,plot=False):
    # Read data
    d = data.read('./data/damping/damping_{:d}_{:d}_{:d}_{:d}.csv'.format(1,int(k*100),int(a*10),n))
    t = np.array(d['t'])
    xm = np.array(d['x'])
    ym = np.array(d['y'])

    # Convert to angle
    # r = np.mean(np.sqrt((xm-xc)**2+(ym-yc)**2)) # 0.0217
    theta = np.arctan2(ym-yc,xm-xc)

    # Select range
    dtheta = np.concatenate([[0],(theta[1:]-theta[:-1])/(t[1:]-t[:-1])])
    i = np.nonzero(dtheta<-5)[0][0]-1
    ip = i

    # Skip the first cycle to avoid unnatural behavior due to manual release
    for ip in range(i,len(dtheta)):
        if dtheta[ip] > 0 and dtheta[ip+1] < 0:
            if dtheta[ip+1] > -5: ip += 1 # Skip flat peak
            break

    j = np.nonzero(t > t[ip]+tfinal)[0][0]
    theta0 = np.mean(theta[j:j+10])
    theta -= theta0

    tp = t[ip:j]
    thetap = theta[ip:j]
    dthetap = dtheta[ip:j]

    if plot:
        plt.figure('theta')
        c = 'C{:d}'.format(n-1)
        plt.plot(t,theta,'-',color=c)
        plt.plot(tp,thetap,'.',color=c)

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
        sol = sim.solve(cs,tf=tfinal)
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

# print(center())
xc,yc = (0.1106286964213686, -0.23876000860024177)

k = 0.1
a = 0.5
tfinal = 0.3 # 0.3, 0.5, 1

bounds=[(0,0.5),(0,0.01)]
x = None

# k=0.1 a=0.5,1,2
x = [0.14171010765472708, 0.00016325042214574825]
# x = [0.12817122286702087, 0.00012936189846465742]
# x = [0.03976104133250624, 2.2544076353877424e-05]

# k=0.2 a=0.5,1,2
# x = [0.15983696179437762, 0.00022822430718920546]
# x = [0.1883647842634908, 0.00015751871012245362]
# x = [0.07235602831789512, 5.742922042662607e-05]

if __name__ == '__main__':
    # for n in [1,2,3]:
    #     read(k,a,n,plot=True)
    # plt.show()
    # exit()

    d = []
    for n in [1,2,3]:
        t_e, theta_e = read(k,a,n)
        d.append([t_e,theta_e])

    if x is None:
        res = differential_evolution(
            obj,
            bounds=bounds,
            args=(d,),
            popsize=20,
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
