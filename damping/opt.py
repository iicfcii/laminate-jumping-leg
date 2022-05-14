from scipy.optimize import minimize,differential_evolution
import matplotlib.pyplot as plt
import numpy as np
from utils import data
from template import jump

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

xc,yc,r = center(plot=False)

k = 0.2
a = 2
n = 1

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
j = np.nonzero(t > t[i]+1)[0][0]
tp = t[i:j]
thetap = theta[i:j]
dthetap = dtheta[i:j]

plt.figure('theta')
plt.plot(t,theta)
plt.plot(tp,thetap)
plt.show()
