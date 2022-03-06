import matplotlib.pyplot as plt
import numpy as np
from . import stiffness

tf = 0.465/1000
tr = np.sum([tf,0.015/1000,0.05/1000,0.015/1000,tf])
wr = 0.01
pad = 0.004 # pad at the end of the flexible beam
rho = 1820 # fiber glass density

# Fourbar
#   b----c
#  /    /
# a----d
def fourbar_fk(bad,ad,ab,bc,cd,form=1):
    l = np.sqrt(ad**2+ab**2-2*ad*ab*np.cos(bad))

    cos_bcd = (bc**2+cd**2-l**2)/(2*bc*cd)
    if np.abs(cos_bcd) > 1 or np.isnan(cos_bcd): return None, None

    sin_bcd = np.sqrt(1-cos_bcd**2)
    if form < 0: sin_bcd = -sin_bcd

    sin_cdb = bc/l*sin_bcd
    cos_cdb = (l**2+cd**2-bc**2)/(2*l*cd)
    cdb = np.arctan2(sin_cdb,cos_cdb)

    sin_adb = ab/l*np.sin(bad)
    cos_adb = (ad**2+l**2-ab**2)/(2*ad*l)
    adb = np.arctan2(sin_adb,cos_adb)

    adc = adb+cdb

    pts = np.array([
        [0,0],
        [ab*np.cos(bad),ab*np.sin(bad)],
        [cd*np.cos(np.pi-adc)+ad,cd*np.sin(np.pi-adc)],
        [ad,0]
    ])

    ts = np.zeros(4)
    for i in range(3):
        ts[i+1] = np.arctan2(pts[i+1,1]-pts[i,1],pts[i+1,0]-pts[i,0])

    return pts, ts

def pose(ps):
    p1 = ps[0,:]
    p2 = ps[1,:]

    center = (p1+p2)/2
    angle = np.arctan2(p2[1]-p1[1],p2[0]-p1[0])
    length = np.linalg.norm(p1-p2)

    return center, angle, length

def limit_ang(ang):
    ang = np.fmod(ang,np.pi*2)
    if ang > np.pi: ang -= 2*np.pi
    if ang < -np.pi: ang += 2*np.pi

    return ang

def leg(ang,l,c,tilt=None):
    ps, ts = fourbar_fk(ang,*l[:4],form=c)
    assert ps is not None, 'No fourbar fk solution'
    pf = np.array([[ps[2,0]+np.cos(ts[2])*l[4],ps[2,1]+np.sin(ts[2])*l[4]]])

    if tilt is None:
        # Make the diagonal line point at -y direction
        tl = np.arctan2(pf[0,1]-ps[0,1],pf[0,0]-ps[0,0])
        tl = -tl-np.pi/2
        tilt = np.array([[np.cos(tl),-np.sin(tl)],[np.sin(tl),np.cos(tl)]])

    ps = (tilt @ ps.T).T
    pf = (tilt @ pf.T).T

    lk = [
        ps[0:2,:],
        ps[1:3,:],
        ps[2:4,:],
        np.array([ps[3,:],ps[0,:]]),
        np.array([ps[2,:],pf[0,:]])
    ]

    return lk, tilt

def spring(ang,ls,c):
    #        b
    #      /  \
    #    d     \
    #   /       \
    # a----------c
    ad = ls[0]
    db = ls[1] # flexible
    bc = ls[2]
    ac = ls[3] # crank
    ab = ad+db

    cos_bac = (ab**2+ac**2-bc**2)/2/ab/ac
    assert np.abs(cos_bac) <= 1, 'Cannot form a triangle'

    # bac within 0 to pi
    bac = np.arccos(cos_bac)
    if c < 0: bac = -bac

    aj = ad+(db-pad)*(1-stiffness.gamma)
    jb = ad+db-aj
    ps, ts = fourbar_fk(-bac+ang,aj,ac,bc,jb,form=-c)
    assert ps is not None, 'No fourbar fk solution'

    lk = [
        ps[0:2,:],
        ps[1:3,:],
        ps[2:4,:],
        np.array([ps[3,:],ps[0,:]]),
    ]

    return lk

def bbox(lks):
    ps = np.array(lks).reshape((-1,2))
    cx = (np.amax(ps[:,0])+np.amin(ps[:,0]))/2
    cy = (np.amax(ps[:,1])+np.amin(ps[:,1]))/2
    pad = 0.005
    dx = np.amax(ps[:,0])-np.amin(ps[:,0])
    dy = np.amax(ps[:,1])-np.amin(ps[:,1])

    return (
        cx-dx/2-pad,cx+dx/2+pad,
        cy-dy/2-pad,cy+dy/2+pad
    )

def leg_spring(ang,l,c,ls,cs):
    lk, tilt = leg(ang,l,c)
    lks = spring(ls,cs)

    ang_crank = pose(lk[0])[1]
    rot = np.array([[np.cos(ang_crank),-np.sin(ang_crank)],[np.sin(ang_crank),np.cos(ang_crank)]])

    lks = [(rot @ ps.T).T for ps in lks]

    return lk+lks
