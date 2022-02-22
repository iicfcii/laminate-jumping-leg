import matplotlib.pyplot as plt
import numpy as np
import geom

# PRBM
gamma = 0.85
Ktheta = 2.65
E = 9.13e9

def prbm_k(t,l,w):
    I = w*t**3/12
    k = gamma*Ktheta*E*I/l
    return k

def sim(x,r,plot=False):
    ls = x[:4]
    w = x[4]
    c = x[5]

    k = prbm_k(0.45/1000,ls[1]-geom.pad,w)
    lk = geom.spring(0,ls,c)

    lks = []
    angs = np.linspace(0,r,50)
    taus = []
    for ang in angs:
        lk = geom.spring(ang,ls,c)
        theta = geom.pose(lk[2])[1]
        alpha = geom.pose(lk[1])[1]
        beta = geom.pose(lk[0])[1]

        assert beta > 10/180*np.pi, 'angle between motor arm and crank too small'

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
        plt.figure()
        plt.axis('scaled')
        plt.xlim([-0.02,0.08])
        plt.ylim([-0.05,0.05])

        for lk in [lks[int(n)] for n in np.linspace(0,len(lks)-1,3)]:
            for link in lk:
                plt.plot(link[:,0],link[:,1],'.-k')

    return angs,taus
