import matplotlib.pyplot as plt
import numpy as np
from . import stiffness

pade = 0.01 # pad at the end of the flexible beam
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

def leg(ang,l,c,tr,tilt=None):
    ps, ts = fourbar_fk(ang,*l[:4],form=c)
    assert ps is not None, 'No fourbar fk solution'

    # beta = 0
    lc = np.sqrt(l[4]**2-tr**2)+np.sqrt(l[2]**2-tr**2)
    cos_beta = (l[4]**2+l[2]**2-lc**2)/2/l[4]/l[2]
    beta = np.arccos(cos_beta) # Consider the joint offset due to beam thickness
    beta = np.pi-beta
    # NOTE: ideally, beta's sign should be determined by testing
    # whether the long coupler intersects with the rocker.
    if c < 0: beta = -beta
    pf = np.array([[
        ps[2,0]+np.cos(ts[2]+beta)*l[4],
        ps[2,1]+np.sin(ts[2]+beta)*l[4]
    ]])

    if tilt is None:
        # Make the diagonal line point at -y direction
        tl = np.arctan2(pf[0,1]-ps[0,1],pf[0,0]-ps[0,0])
        tl = -tl-np.pi/2
        tilt = np.array([[np.cos(tl),-np.sin(tl)],[np.sin(tl),np.cos(tl)]])

    ps = (tilt @ ps.T).T
    pf = (tilt @ pf.T).T

    lk = [
        ps[0:2,:], # crank
        ps[1:3,:], # coupler
        ps[2:4,:], # rocker
        np.array([ps[3,:],ps[0,:]]), # ground
        np.array([ps[2,:],pf[0,:]]) # foot
    ]

    return lk, tilt

def spring(ang,ls,c):
    #        b
    #      /  \
    #    j     \
    #   /       \
    # a----------c
    padf = ls[0] # pad at the front of the flexible beam
    ab = padf+ls[1]+pade # flexible
    bc = ls[2]
    ac = ls[3] # crank

    cos_bac = (ab**2+ac**2-bc**2)/2/ab/ac
    assert np.abs(cos_bac) <= 1, 'Cannot form a triangle'

    # bac within 0 to pi
    bac = np.arccos(cos_bac)
    if c < 0: bac = -bac

    aj = padf+ls[1]*(1-stiffness.gamma)
    jb = ab-aj
    ps, ts = fourbar_fk(-bac+ang,aj,ac,bc,jb,form=-c)
    assert ps is not None, 'No fourbar fk solution'

    lk = [
        ps[0:2,:], # ac, crank
        ps[1:3,:], # bc, coupler
        ps[2:4,:], # jb, coupler
        np.array([ps[3,:],ps[0,:]]), # aj, ground
    ]

    return lk

def bbox(lks,pad=0.005):
    ps = np.array(lks).reshape((-1,2))
    cx = (np.amax(ps[:,0])+np.amin(ps[:,0]))/2
    cy = (np.amax(ps[:,1])+np.amin(ps[:,1]))/2
    dx = np.amax(ps[:,0])-np.amin(ps[:,0])
    dy = np.amax(ps[:,1])-np.amin(ps[:,1])

    return (
        cx-dx/2-pad,cx+dx/2+pad,
        cy-dy/2-pad,cy+dy/2+pad
    )

def leg_spring(ang,l,c,ls,cs,tr):
    lk, tilt = leg(ang,l,c,tr)
    lks = spring(0,ls,cs)

    ang_crank = pose(lk[0])[1]
    ang_crankp = pose(lks[0])[1]
    dang = ang_crank-ang_crankp
    rot = np.array([[np.cos(dang),-np.sin(dang)],[np.sin(dang),np.cos(dang)]])

    lks = [(rot @ ps.T).T for ps in lks]

    return lk+lks
