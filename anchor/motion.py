import matplotlib.pyplot as plt
import numpy as np
from . import geom

tr = np.sum([0.83,0.015,0.05,0.015,0.45])/1000
wr = 0.01

def sim(rots,x,plot=False):
    ang = x[0]
    l = x[1:6]
    c = x[6]

    lk, tilt = geom.leg(ang,l,c,tr)

    lks = []
    for rot in rots:
        lk, rot = geom.leg(rot,l,c,tr,tilt=tilt)
        lks.append(lk)

    lks = np.array(lks)
    xs = lks[:,4,1,0]
    ys = lks[:,4,1,1]

    if plot:
        lks_r = [lks[int(n)] for n in np.linspace(0,len(lks)-1,3)]
        bbox = geom.bbox(lks_r)
        plt.figure()
        plt.axis('scaled')
        plt.xlim(bbox[:2])
        plt.ylim(bbox[2:])
        for lk in lks_r:
            for link in lk:
                plt.plot(link[:,0],link[:,1],'k')

        pfs = lks[:,4,1,:]
        plt.plot(pfs[:,0],pfs[:,1])

    return xs, ys

def simf(rots,x,plot=False):
    ang = x[0]
    l = x[1:6]
    c = x[6]

    lk, tilt = geom.leg(ang,l,c,tr)

    rs = []
    for rot in rots:
        lk, rotp = geom.leg(rot,l,c,tr,tilt=tilt)

        fy = 1
        mu = 0
        fx = fy*mu
        ang_fr = geom.pose(lk[2])[1]
        dx_fr = lk[2][0,0]-lk[1][0,0]
        dy_fr = lk[2][0,1]-lk[1][0,1]
        dx_f = lk[4][1,0]-lk[1][0,0]
        dy_f = lk[4][1,1]-lk[1][0,1]
        fr = -(dx_f*fy-dy_f*fx)/(dx_fr*np.sin(ang_fr)-dy_fr*np.cos(ang_fr))

        fiy = -fy-fr*np.sin(ang_fr)
        fix = -fx-fr*np.cos(ang_fr)

        ang_crank = geom.pose(lk[0])[1]
        l_crank = l[1]
        tau = -(l_crank*np.cos(ang_crank)*fiy-l_crank*np.sin(ang_crank)*fix)

        rs.append(abs(tau/fy))
    rs = np.array(rs)

    if plot:
        plt.figure()
        plt.plot(rots,rs)

    return rs
