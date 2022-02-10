import sys
sys.path.append('../spring')

import pychrono as chrono
import pychrono.irrlicht as chronoirr
import matplotlib.pyplot as plt
import numpy as np
import prbm

import os.path
chrono.SetChronoDataPath(os.path.join(os.path.abspath('../chrono_data/'),''))

tf = 16.5*2.54e-5
wr = 0.02
tr = np.sum([0.4191,0.015,0.0508,0.015,0.4191])/1000
rho = prbm.rho
pad = 0 # pad at ends for wider hinge
step = 5e-5
tfinal = 1

# Fourbar
#   b----c
#  /    /
# a----d
def fk(bad,ad,ab,bc,cd,form=1):
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

def leg(ang,l,c,tilt=None):
    ps, ts = fk(ang,*l[:4],form=c)
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

def spring(ls,c):
    #        b
    #      /  \
    #    d     \
    #   /       \
    # a----------c
    ad = ls[0]
    db = ls[1] # Flexible
    bc = ls[2]
    ac = ls[3]
    ab = ad+db


    cos_bac = (ab**2+ac**2-bc**2)/2/ab/ac
    assert np.abs(cos_bac) <= 1, 'Cannot form a triangle'

    # bac within 0 to pi
    bac = np.arccos(cos_bac)
    if c < 0: bac = -bac

    a = np.array([0,0])
    b = np.array([ab*np.cos(bac),ab*np.sin(bac)])
    c = np.array([ac,0])

    j = (b-a)*((1-ad/ab)*(1-prbm.gamma)+ad/ab)

    lks = [
        np.array([c,a]),
        np.array([a,j]),
        np.array([j,b]),
        np.array([b,c])
    ]

    return lks

def motion(rots,x,plot=False):
    ang = x[0]
    l = x[1:6]
    c = x[6]

    lk, tilt = leg(ang,l,c)

    lks = []
    for rot in rots:
        lk, rot = leg(rot,l,c,tilt=tilt)
        lks.append(lk)

    lks = np.array(lks)
    xs = lks[:,4,1,0]
    ys = lks[:,4,1,1]

    if plot:
        plt.figure()
        plt.axis('scaled')
        plt.xlim([-0.06,0.06])
        plt.ylim([-0.12,0.01])
        for lk in lks:
            for link in lk:
                plt.plot(link[:,0],link[:,1],'k',linewidth=0.5)

        pfs = lks[:,4,1,:]
        plt.plot(pfs[:,0],pfs[:,1])

    return xs, ys

springTorques = []
class RotSpringTorque(chrono.TorqueFunctor):
    def __init__(self, k, b):
        super(RotSpringTorque, self).__init__()
        self.k = k
        self.b = b

    def __call__(self,time,angle,vel,link):
        torque = -self.k*angle-self.b*vel
        return torque

def stiffness(x,cs,plot=False):
    ls = x[:4]
    w = x[4]
    c = x[5]

    lk = spring(ls,c)

    system = chrono.ChSystemNSC()
    system.Set_G_acc(chrono.ChVectorD(0,0,0))

    links = []
    for i in range(len(lk)):
        if i == 1 or i == 2:
            tl = tf
            wl = w
        else:
            tl = tr
            wl = wr

        pos, rot, length = pose(lk[i])

        link = chrono.ChBodyEasyBox(length,tl,wl,rho,True)
        link.SetPos(chrono.ChVectorD(*pos,0))
        link.SetRot(chrono.Q_from_AngZ(rot))
        system.Add(link)
        links.append(link)
    links[1].SetBodyFixed(True)

    # Joints and Springs
    joints = []
    springs = []
    springTorques.clear()
    for i in range(len(links)):
        if i == 0:
            bodies = [links[1]]
            fixed = [False]
        elif i == 1:
            bodies = [links[2]]
            fixed = [False]
        elif i == 2:
            bodies = [links[3]]
            fixed = [False]
        elif i == 3:
            bodies = [links[0]]
            fixed = [False]
        else:
            bodies = []
            fixed = []

        pos = lk[i][1,:]

        if i == 1:
            k = prbm.k(tf,ls[1],w)
        else:
            k = 0

        b = 0

        for bd,f in zip(bodies,fixed):
            joint = chrono.ChLinkMateGeneric(True,True,True,True,True,f)
            joint.Initialize(links[i],bd,chrono.ChFrameD(chrono.ChVectorD(*pos,0)))
            system.Add(joint)
            joints.append(joint)

            if f: continue

            try:
                spring_link = chrono.ChLinkRotSpringCB()
                spring_link.Initialize(links[i],bd,chrono.ChCoordsysD(chrono.ChVectorD(*pos,0)))
                springTorques.append(RotSpringTorque(k,b))
                spring_link.RegisterTorqueFunctor(springTorques[-1])
                system.AddLink(spring_link)
                springs.append(spring_link)
            except AttributeError:
                # Accommodate develop version of chrono
                spring_link = chrono.ChLinkRSDA()
                spring_link.Initialize(links[i],bd,chrono.ChCoordsysD(chrono.ChVectorD(*pos,0)))
                spring_link.SetSpringCoefficient(k)
                spring_link.SetDampingCoefficient(b)
                system.AddLink(spring_link)
                springs.append(spring_link)

    motor = chrono.ChLinkMotorRotationAngle()
    motor.Initialize(
        links[1],
        links[0],
        chrono.ChFrameD(chrono.ChVectorD(0,0,0))
    )
    motorTorque = chrono.ChFunction_Sine(0,1/tfinal/2,-cs['dsm']/cs['r'])
    motor.SetAngleFunction(motorTorque)
    system.Add(motor)

    datum = {
        'x': [],
        'f': [],
    }
    def record():
        datum['x'].append(motor.GetMotorRot())
        datum['f'].append(motor.GetMotorTorque())

    if plot:
        application = chronoirr.ChIrrApp(system, "Jump", chronoirr.dimension2du(800, 600),chronoirr.VerticalDir_Y)
        application.AddTypicalSky()
        application.AddTypicalLights()
        y_offset = 0
        z_offset = -0.15
        application.AddTypicalCamera(chronoirr.vector3df(0, y_offset, z_offset),chronoirr.vector3df(0, y_offset, 0))
        application.AssetBindAll()
        application.AssetUpdateAll()

        application.SetTimestep(step)
        # application.SetVideoframeSaveInterval(int(1/step/2000))
        # application.SetVideoframeSave(True)

        while application.GetDevice().run():
            record()
            application.BeginScene()
            application.DrawAll()

            # Draw axis for scale and orientation
            s = 0.1
            chronoirr.drawSegment(application.GetVideoDriver(),chrono.ChVectorD(0,0,0),chrono.ChVectorD(s,0,0),chronoirr.SColor(1,255,0,0))
            chronoirr.drawSegment(application.GetVideoDriver(),chrono.ChVectorD(0,0,0),chrono.ChVectorD(0,s,0),chronoirr.SColor(1,0,255,0))
            chronoirr.drawSegment(application.GetVideoDriver(),chrono.ChVectorD(0,0,0),chrono.ChVectorD(0,0,s),chronoirr.SColor(1,0,0,255))

            application.DoStep()
            application.EndScene()

            if system.GetChTime() > tfinal: application.GetDevice().closeDevice()

        # plt.figure()
        # plt.axis('scaled')
        # r = 0.12
        # plt.xlim([-r,r])
        # plt.ylim([-r,r])
        # for link in lk:
        #     plt.plot(link[:,0],link[:,1],'.-k')
    else:
        system.SetChTime(0)
        while True:
            record()
            system.DoStepDynamics(step)
            if system.GetChTime() > tfinal: break

    return datum
