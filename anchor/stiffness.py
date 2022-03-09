import matplotlib.pyplot as plt
import numpy as np
from . import geom

# PRBM
gamma = 0.85
Ktheta = 2.65
# E (GPa) vs l(mm) 3 deg poly
pE = [1.8117485942295127e-05, -0.0044047993719374965, 0.3702293952041017, 5.703030702032034]

def prbm_k(t,l,w):
    I = w*t**3/12
    E = np.polyval(pE,l*1000)*1e9
    k = gamma*Ktheta*E*I/l
    return k

def sim(x,r,plot=False):
    ls = x[:4]
    w = x[4]
    c = x[5]

    k = prbm_k(geom.tf,ls[1]-geom.pad,w)
    lk = geom.spring(0,ls,c)

    lks = []
    angs = np.linspace(0,r,50)
    taus = []
    for ang in angs:
        lk = geom.spring(ang,ls,c)
        theta = geom.pose(lk[2])[1]
        alpha = geom.pose(lk[1])[1]
        beta = geom.pose(lk[0])[1]

        assert np.abs(beta) > 20/180*np.pi, 'angle between motor arm and crank too small'

        # Static analysis
        dtheta = geom.limit_ang(theta-np.pi)
        tauk = k*dtheta
        f_ang = geom.limit_ang(alpha-theta)
        f = tauk/(ls[1]*gamma)/np.sin(f_ang)
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
