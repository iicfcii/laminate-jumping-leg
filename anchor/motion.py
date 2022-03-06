import matplotlib.pyplot as plt
import numpy as np
from . import geom

def sim(rots,x,plot=False):
    ang = x[0]
    l = x[1:6]
    c = x[6]

    lk, tilt = geom.leg(ang,l,c)

    lks = []
    for rot in rots:
        lk, rot = geom.leg(rot,l,c,tilt=tilt)
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
