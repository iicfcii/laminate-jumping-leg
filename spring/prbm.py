import sys
sys.path.append('../utils')

from scipy.optimize import root_scalar
import numpy as np
import process
import data

gamma = 0.85
Ktheta = 2.65
E = 18.6e9*1.1

def k(t,l,w):
    I = w*t**3/12
    k = gamma*Ktheta*E*I/l
    return k

def sim(rot,tz,t,l,w):
    t = tmil*2.54e-5
    w = wmm/1000
    l = lmm/1000

    K = k(t,l,w)

    def eq(theta):
        return tz/l*np.cos(theta-rot)*gamma*l-K*theta
    sol = root_scalar(eq, bracket=[-np.pi/2, np.pi/2], method='brentq')
    assert sol.converged
    theta = sol.root
    # theta = tz/l*gamma*l/K
    x = np.cos(theta)*gamma*l+(1-gamma)*l
    y = np.sin(theta)*gamma*l

    return x,y

if __name__ == '__main__':
    for tmil in [16.5,32.5]:
        for wmm in [10,20,30]:
            for lmm in [25,50,75]:
                rot, tz = process.sample(tmil,lmm,wmm,'1',np.pi/4 if tmil==32.5 else 0)

                x = []
                y = []
                for r,t in zip(rot,tz):
                    pos = sim(r,t,tmil,lmm,wmm)
                    x.append(pos[0])
                    y.append(pos[1])

                file_name = '../data/{:d}mil_{:d}mm_{:d}mm_prbm.csv'.format(int(tmil/5)*5,int(lmm),wmm)
                data.write(
                    file_name,
                    ['x','y','rot','tz'],
                    [x,y,rot,tz]
                )

                print(tmil,lmm,wmm,'done')
