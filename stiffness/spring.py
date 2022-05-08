import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from utils import data
from anchor import stiffness
from template import jump

STEP = 1 # s
KEYS = ['rz','fx','fy','tz']
SELECT = {
    'spring': ['tz',0.001],
    'leg': ['fx',0.05]
}

def read(s,k,a,n,type='leg'):
    name = './data/spring/{}_{:d}_{:d}_{:d}_{:d}.csv'.format(type,s,int(k*100),int(a*10),n)
    d = data.read(name)
    t = np.array(d['t'])

    # Avoid jump between -pi and pi
    rz = np.array(d['rz'])
    rz[rz < 0] += 2*np.pi
    d['rz'] = rz

    # Remove bias
    dp = {}
    for k in KEYS:
        f = np.array(d[k])
        offset = np.average(f[np.logical_and(t>1,t<3)])
        f -= offset
        f = np.sign(f[-1])*f
        dp[k] = f

    # Take mean
    tm = []
    dpm = {}
    tc = 1
    while tc < t[-1]:
        idx = np.logical_and(t>tc-STEP/5,t<tc+STEP/5)
        tm.append(tc)
        for k in KEYS:
            v = np.mean(dp[k][idx])
            if k not in dpm:
                dpm[k] = [v]
            else:
                dpm[k].append(v)
        tc += 1

    # Convert to numpy array
    tm = np.array(tm)
    for k in KEYS:
        dpm[k] = np.array(dpm[k])

    # plt.figure('rz')
    # plt.plot(t,dp['rz'],color='C0',linewidth=1)
    # plt.plot(tm,dpm['rz'],'.',color='C0',markersize=10)
    #
    # plt.figure('tz')
    # plt.plot(t,dp['tz'],color='C0',linewidth=1)
    # plt.plot(tm,dpm['tz'],'.',color='C0',markersize=10)
    #
    # plt.figure('k')
    # plt.plot(dpm['rz'],dpm['tz'],'.')
    # plt.show()

    k,th = SELECT[type]
    i = np.nonzero(dpm[k] > th)[0][0]-1
    rzm = dpm['rz'][i:]
    rzm -= rzm[0]
    fm = dpm[k][i:]
    fm -= fm[0]

    return rzm,fm

def readn(s,k,a,type='leg'):
    rzs = []
    fs = []
    for n in [1,2,3]:
        rz,f = read(s,k,a,n,type=type)
        rzs.append(rz)
        fs.append(f)

    rzi = np.arange(0,jump.cs['t']/k,0.01)

    fis = []
    for rz,f in zip(rzs,fs):
        fi = np.interp(rzi,rz,f)
        fis.append(fi)
    fi = np.mean(fis,axis=0)

    # for rz,f in zip(rzs,fs):
    #     plt.plot(rz,f,'.')
    # plt.plot(rzi,fi)
    # plt.show()

    return rzi,fi,rzs,fs

def fit(rz,tz,kp,ap):
    def obj(rz,k,a):
        return jump.t_spring(rz,k,a,jump.cs['t'])

    popt, pcov = curve_fit(obj,rz,tz,p0=[kp,ap])
    k,a = popt

    return k,a

if __name__ == '__main__':
    plt.figure()
    for k in [0.1,0.2]:
        for a in [0.5,1,2]:
            for s in [1]:
                for i,n in enumerate([1,2,3]):
                    rz,f = read(s,k,a,n,type='leg')
                    # f = f*jump.cs['r']
                    f = f*0.032

                    c = 'C{:d}'.format(i)
                    plt.plot(rz,f,'.-',markersize=4,color=c)

                    kp,ap = fit(rz,f,k,a)
                    print(kp,ap)

            rzp = np.linspace(0,jump.cs['t']/k,100)
            fp = jump.t_spring(rzp,k,a,jump.cs['t'])
            plt.plot(rzp,fp,'--k')
    plt.show()
