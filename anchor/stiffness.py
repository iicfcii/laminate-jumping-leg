import matplotlib.pyplot as plt
import numpy as np
from . import geom

# PRBM
t_30 = 0.83
t_15 = 0.45
t_10 = 0.27
ang_min = 15/180*np.pi
tr = np.sum([t_30,0.015,0.05,0.015])/1000
wr = 0.01
gamma = 0.85
Ktheta = 2.65
E = 1.5*1e6*6894.76

def prbm_k(t,l,w):
    I = w*t**3/12
    k = gamma*Ktheta*E*I/l
    return k

def tf(ct):
    return t_15/1000 if ct > 0 else t_30/1000

def sim(x,r,plot=False):
    ls = x[:4]
    cm = x[4]
    ct = x[5]
    wf = x[6]

    k = prbm_k(tf(ct),ls[1],wf)
    lk = geom.spring(0,ls,cm)

    lks = []
    angs = np.linspace(0,r,50)
    taus = []
    for ang in angs:
        lk = geom.spring(ang,ls,cm)
        theta = geom.pose(lk[2])[1]
        alpha = geom.pose(lk[1])[1]
        beta = geom.pose(lk[0])[1]
        assert np.abs(beta) > ang_min, 'angle between motor arm and crank too small'

        # Static analysis
        dtheta = geom.limit_ang(theta-np.pi)
        tauk = k*dtheta
        f_ang = geom.limit_ang(alpha-theta)
        assert np.abs(geom.limit_ang(np.pi-f_ang)) > ang_min, 'angle between flexible beam and coupler too small'
        f = tauk/(ls[1]*gamma+geom.pade)/np.sin(f_ang)
        fp_ang = geom.limit_ang(np.pi+alpha-beta)
        tau = f*np.sin(fp_ang)*ls[3]

        # print(theta,alpha,beta,dtheta,f_ang,fp_ang)
        taus.append(tau)
        lks.append(lk)

    angs = np.array(angs)
    taus = np.array(taus)

    if plot:
        lks_r = [lks[int(n)] for n in np.linspace(0,len(lks)-1,3)]
        bbox = geom.bbox(lks_r)
        plt.figure()
        plt.axis('scaled')
        plt.xlim(bbox[:2])
        plt.ylim(bbox[2:])

        for lk in lks_r:
            for link in lk:
                plt.plot(link[:,0],link[:,1],'.-k')

    return angs,taus
