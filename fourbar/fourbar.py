import sys
sys.path.append('../spring')

import pychrono as chrono
import pychrono.irrlicht as chronoirr
import matplotlib.pyplot as plt
import numpy as np
import prbm

import os.path
chrono.SetChronoDataPath(os.path.join(os.path.abspath('../chrono_data/'),''))

pi = np.pi
wr = 0.02
tr = 0.0013
tf = 32.5*2.54e-5
rho = prbm.rho

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
        [cd*np.cos(pi-adc)+ad,cd*np.sin(pi-adc)],
        [ad,0]
    ])

    ts = np.zeros(4)
    for i in range(3):
        ts[i+1] = np.arctan2(pts[i+1,1]-pts[i,1],pts[i+1,0]-pts[i,0])

    return pts, ts

# l: ground crank coupler rocker ext
# w: crank ext
# c: fourbar cross form
# dir: crank dir
# m: body mass including motor, circuits, connectors
def solve(ang,l,w,c,m,cs,vis=False):
    step = 5e-6
    tfinal = 0.2

    # Forward kinematics
    ps, ts = fk(ang,*l[:4],form=c)
    assert ps is not None, 'No fourbar fk solution'
    pf = np.array([[ps[2,0]+np.cos(ts[2])*l[4],ps[2,1]+np.sin(ts[2])*l[4]]])

    # Make the diagonal line point at -y direction
    tl = np.arctan2(pf[0,1]-ps[0,1],pf[0,0]-ps[0,0])
    tl = -tl-pi/2
    rot = np.array([[np.cos(tl),-np.sin(tl)],[np.sin(tl),np.cos(tl)]])
    ps = (rot @ ps.T).T
    pf = (rot @ pf.T).T

    ls = []
    # Ground
    p1 = ps[3,:]
    p2 = ps[0,:]
    ls.append(np.array([p1,p2]))
    # Crank
    p1 = ps[0,:]
    p2 = ps[1,:]
    pad = 0.005 # 5mm pad at both ends for wider hinge
    pj = (p2-p1)*((1-pad*2/l[1])*(1-prbm.gamma)+pad/l[1])+p1
    ls.append(np.array([p1,pj]))
    ls.append(np.array([pj,p2]))
    # Coupler
    p1 = ps[1,:]
    p2 = ps[2,:]
    ls.append(np.array([p1,p2]))
    # Ext
    p1 = ps[2,:]
    p2 = pf[0,:]
    pj = (p2-p1)*(1-prbm.gamma)+p1
    ls.append(np.array([p1,pj]))
    ls.append(np.array([pj,p2]))
    # Rocker
    p1 = ps[2,:]
    p2 = ps[3,:]
    ls.append(np.array([p1,p2]))

    # plt.figure()
    # for link in ls:
    #     plt.plot(link[:,0],link[:,1],'-ok')
    # plt.axis('scaled')
    # r = 0.15
    # plt.xlim([-r,r])
    # plt.ylim([-r,r])
    # plt.show()

    def pose(ps):
        p1 = ps[0,:]
        p2 = ps[1,:]

        center = (p1+p2)/2
        angle = np.arctan2(p2[1]-p1[1],p2[0]-p1[0])
        length = np.linalg.norm(p1-p2)

        return center, angle, length

    system = chrono.ChSystemNSC()
    system.Set_G_acc(chrono.ChVectorD(0,-9.81,0))

    ground = chrono.ChBodyEasyBox(0.1,tr,wr,rho,True)
    ground.SetPos(chrono.ChVectorD(*pf.flatten(),0))
    ground.SetRot(chrono.Q_from_AngZ(0))
    ground.SetBodyFixed(True)
    system.Add(ground)

    wb = (m/rho)**(1/3)
    body = chrono.ChBodyEasyBox(wb,wb,wb,rho,False)
    body.SetPos(chrono.ChVectorD(0,0,0))
    body.SetRot(chrono.Q_from_AngZ(0))
    system.Add(body)

    links = []
    # Bodies
    for i in range(len(ls)):
        pos, rot, length = pose(ls[i])
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
        p = ls[l1_i][1,:]

        if i != 3:
            joint = chrono.ChLinkMateGeneric(True,True,True,True,True,False)
        else: # coupler to start of ext
            joint = chrono.ChLinkMateGeneric(True,True,True,True,True,True)
        joint.Initialize(l1,l2,chrono.ChFrameD(chrono.ChVectorD(*p,0)))
        system.Add(joint)
        joints.append(joint)

        if i == 1:
            kl = prbm.k(tf,l[1],w[0])
        elif i == 4:
            kl = prbm.k(tf,l[4],w[1])
        else:
            kl = 0.0001

        if i == 0:
            b = cs['tau']/(cs['v'])
        else:
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

    joint_vertical = chrono.ChLinkMateGeneric(True,False,True,True,True,True)
    joint_vertical.Initialize(ground,body,chrono.ChFrameD(chrono.ChVectorD(0,0,0)))
    system.Add(joint_vertical)

    joint_link = chrono.ChLinkMateGeneric(True,True,True,True,True,True)
    joint_link.Initialize(body,links[0],chrono.ChFrameD(chrono.ChVectorD(0,0,0)))
    system.Add(joint_link)

    joint_foot = chrono.ChLinkMateGeneric(True,True,True,True,True,False)
    joint_foot.Initialize(ground,links[5],chrono.ChFrameD(chrono.ChVectorD(*pf.flatten(),0)))
    system.Add(joint_foot)

    class MotorTorque(chrono.ChFunction):
        def __init__(self, body):
            super().__init__()
            self.body = body

        def Get_y(self, t):
            y = self.body.GetPos().y
            return (np.maximum(0,y-cs['em'])/(y-cs['em'])-1)*cs['tau']

    motor = chrono.ChLinkMotorRotationTorque()
    motor.Initialize(links[1],links[0],chrono.ChFrameD(chrono.ChVectorD(0,0,0)))
    motorTorque = MotorTorque(body)
    motor.SetTorqueFunction(motorTorque)
    system.Add(motor)

    data = {
        't': [],
        'fx': [],
        'fy': [],
        'fxb': [],
        'tzb': [],
        'yb':[],
        'dyb':[],
    }
    def record():
        data['t'].append(system.GetChTime())
        data['fx'].append(joint_foot.Get_react_force().x)
        data['fy'].append(joint_foot.Get_react_force().y)

        data['fxb'].append(joint_vertical.Get_react_force().x)
        data['tzb'].append(joint_vertical.Get_react_torque().z)

        data['yb'].append(body.GetPos().y)
        data['dyb'].append(body.GetPos_dt().y)

    def end():
        # End points of every link should not be below ground
        collide = False
        for i,link in enumerate(links):
            pos = [link.GetPos().x,link.GetPos().y]
            rot = link.GetRot().Q_to_Euler123().z
            length = np.linalg.norm(ls[i][1,:]-ls[i][0,:])

            y1 = pos[1]+length/2*np.sin(rot)
            y2 = pos[1]-length/2*np.sin(rot)

            eps = 1e-3
            if y1 < pf[0,1]-eps or y2 < pf[0,1]-eps:
                collide = True

        return (
            (joint_foot.Get_react_force().y < 1e-6 and system.GetChTime() > 1e-2) or
            collide or
            system.GetChTime() > tfinal
        )

    if vis:
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

            if end(): application.GetDevice().closeDevice()
    else:
        system.SetChTime(0)
        while True:
            record()
            system.DoStepDynamics(step)
            if end(): break

    return data
