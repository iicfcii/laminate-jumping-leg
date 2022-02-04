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
pad = 0.006 # 6mm pad at ends for wider hinge

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

def leg(ang,l,c,tilt=None,is_prbm=False):
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

    if not is_prbm:
        links = [
            ps[0:2,:],
            ps[1:3,:],
            ps[2:4,:],
            np.array([ps[3,:],ps[0,:]]),
            np.array([ps[2,:],pf[0,:]])
        ]
    else:
        links = []
        # Ground
        p1 = ps[3,:]
        p2 = ps[0,:]
        links.append(np.array([p1,p2]))
        # Crank
        p1 = ps[0,:]
        p2 = ps[1,:]
        pj = (p2-p1)*((1-pad*2/l[1])*(1-prbm.gamma)+pad/l[1])+p1
        links.append(np.array([p1,pj]))
        links.append(np.array([pj,p2]))
        # Coupler
        p1 = ps[1,:]
        p2 = ps[2,:]
        links.append(np.array([p1,p2]))
        # Ext
        p1 = ps[2,:]
        p2 = pf[0,:]
        pj = (p2-p1)*((1-pad/l[4])*(1-prbm.gamma)+pad/l[4])+p1
        links.append(np.array([p1,pj]))
        links.append(np.array([pj,p2]))
        # Rocker
        p1 = ps[2,:]
        p2 = ps[3,:]
        links.append(np.array([p1,p2]))

    return links, tilt

def motion(rots,l,c,plot=False):
    lk, tilt = leg(rots[0],l,c)

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
        r = 0.12
        plt.xlim([-r,r])
        plt.ylim([-r,r])
        for lk in lks:
            for link in lk:
                plt.plot(link[:,0],link[:,1],'k',linewidth=0.5)

        pfs = lks[:,4,1,:]
        plt.plot(pfs[:,0],pfs[:,1])

    return xs, ys

def spring(rots,l,c,w,ds,plot=False):
    step = 10e-5
    tfinal = 1
    tilt = leg(rots[0],l,c,is_prbm=True)[1]

    def pose(ps):
        p1 = ps[0,:]
        p2 = ps[1,:]

        center = (p1+p2)/2
        angle = np.arctan2(p2[1]-p1[1],p2[0]-p1[0])
        length = np.linalg.norm(p1-p2)

        return center, angle, length

    def sim(rot,l,c,w):
        lk = leg(rot,l,c,tilt=tilt,is_prbm=True)[0]
        system = chrono.ChSystemNSC()
        system.Set_G_acc(chrono.ChVectorD(0,0,0))

        links = []
        for i in range(len(lk)):
            pos, rot, length = pose(lk[i])
            if i == 1 or i == 2:
                tl = tf
                wl = w[0]
            elif i == 4 or i == 5:
                tl = tf
                wl = w[1]
            else:
                tl = tr
                wl = wr
            link = chrono.ChBodyEasyBox(length,tl,wl,rho,True)
            link.SetPos(chrono.ChVectorD(*pos,0))
            link.SetRot(chrono.Q_from_AngZ(rot))
            system.Add(link)
            links.append(link)

        links[0].SetBodyFixed(True)
        links[1].SetBodyFixed(True)

        # Joints and Springs
        class RotSpringTorque(chrono.TorqueFunctor):
            def __init__(self, k, b):
                super(RotSpringTorque, self).__init__()
                self.k = k
                self.b = b

            def __call__(self,time,angle,vel,link):
                torque = -self.k*angle-self.b*vel
                return torque

        joints = []
        springs = []
        springTorques = []
        for i in range(len(links)):
            if i == 6: # rocker to ground
                l1_i = i
                l2_i = 0
            elif i == 5: # rocker to coupler
                l1_i = 3
                l2_i = 6
            else:
                l1_i = i
                l2_i = i+1

            l1 = links[l1_i]
            l2 = links[l2_i]
            p = lk[l1_i][1,:]

            if i != 3:
                joint = chrono.ChLinkMateGeneric(True,True,True,True,True,False)
            else: # coupler to start of ext
                joint = chrono.ChLinkMateGeneric(True,True,True,True,True,True)
            joint.Initialize(l1,l2,chrono.ChFrameD(chrono.ChVectorD(*p,0)))
            system.Add(joint)
            joints.append(joint)

            if i == 1:
                kl = prbm.k(tf,l[1]-pad*2,w[0])
            elif i == 4:
                kl = prbm.k(tf,l[4]-pad,w[1])
            else:
                kl = 0

            b = 0

            try:
                spring = chrono.ChLinkRotSpringCB()
                spring.Initialize(l1,l2,chrono.ChCoordsysD(chrono.ChVectorD(*p,0)))
                springTorques.append(RotSpringTorque(kl,b))
                spring.RegisterTorqueFunctor(springTorques[i])
                system.AddLink(spring)
                springs.append(spring)
            except AttributeError:
                # Accommodate develop version of chrono
                spring = chrono.ChLinkRSDA()
                spring.Initialize(l1,l2,chrono.ChCoordsysD(chrono.ChVectorD(*p,0)))
                spring.SetSpringCoefficient(kl)
                spring.SetDampingCoefficient(b)
                system.AddLink(spring)
                springs.append(spring)

        motor = chrono.ChLinkMotorLinearPosition()
        motor.Initialize(
            links[0],
            links[5],
            chrono.ChFrameD(chrono.ChVectorD(*lk[5][1,:],0),chrono.Q_from_AngZ(np.pi/2))
        )
        motorTorque = chrono.ChFunction_Sine(0,1/tfinal,ds)
        motor.SetMotionFunction(motorTorque)
        system.Add(motor)

        datum = {
            'x': [],
            'f': [],
        }
        def record():
            datum['x'].append(motor.GetMotorPos())
            datum['f'].append(motor.GetMotorForce())

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
                l = 0.1
                chronoirr.drawSegment(application.GetVideoDriver(),chrono.ChVectorD(0,0,0),chrono.ChVectorD(l,0,0),chronoirr.SColor(1,255,0,0))
                chronoirr.drawSegment(application.GetVideoDriver(),chrono.ChVectorD(0,0,0),chrono.ChVectorD(0,l,0),chronoirr.SColor(1,0,255,0))
                chronoirr.drawSegment(application.GetVideoDriver(),chrono.ChVectorD(0,0,0),chrono.ChVectorD(0,0,l),chronoirr.SColor(1,0,0,255))

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

    data = []
    for rot in rots:
        datum = sim(rot,l,c,w)
        data.append(datum)

    return data
