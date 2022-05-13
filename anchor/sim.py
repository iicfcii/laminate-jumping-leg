import pychrono as chrono
import pychrono.irrlicht as chronoirr
import matplotlib.pyplot as plt
import numpy as np
from anchor import geom, design, motion, stiffness
from template import jump

import os.path
chrono.SetChronoDataPath(os.path.join(os.path.abspath('./chrono_data/'),''))

springTorques = []
class RotSpringTorque(chrono.TorqueFunctor):
    def __init__(self, k, b):
        super(RotSpringTorque, self).__init__()
        self.k = k
        self.b = b

    def __call__(self,time,angle,vel,link):
        torque = -self.k*angle-self.b*vel
        return torque

class MotorTorqueDC(chrono.ChFunction):
    def __init__(self,ground,crank):
        super().__init__()
        self.ground = ground
        self.crank = crank

        rg = self.ground.GetRot().Q_to_Euler123().z
        rc = self.crank.GetRot().Q_to_Euler123().z

        self.i = 0

    def Get_y(self, t):
        # rg = self.ground.GetRot().Q_to_Euler123().z-np.pi
        wg = chrono.ChVectorD()
        self.ground.GetRot_dt().Qdt_to_Wabs(wg,self.ground.GetRot())

        # rc = self.crank.GetRot().Q_to_Euler123().z
        wc = chrono.ChVectorD()
        self.crank.GetRot_dt().Qdt_to_Wabs(wc,self.crank.GetRot())

        dtheta = wc.z-wg.z

        if t > 0.1:
            di = (-jump.cs['V']-jump.cs['K']*dtheta-jump.cs['R']*self.i)/jump.cs['L']
            self.i = di*step+self.i
            torque = jump.cs['K']*self.i-jump.cs['b']*dtheta
            # print(t,dtheta,self.i,torque)
        else:
            torque = 0

        return torque

tfinal = 0.5
step = 1e-5

def run(idx):
    xm = design.xm
    ks = design.springs[idx]['k']
    xs = design.springs[idx]['x']
    # print(xm)
    # print(xs)

    ang = xm[0]
    l = xm[1:6]
    c = xm[6]
    ls = xs[:4]
    cm = xs[4]
    ct = xs[5]
    wf = xs[6]
    lk = geom.leg_spring(ang,l,c,ls,cm,motion.tr)

    system = chrono.ChSystemNSC()
    system.Set_G_acc(chrono.ChVectorD(0,-9.81*1,0))

    chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.001)
    chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.001)
    contact_mat = chrono.ChMaterialSurfaceNSC()
    contact_mat.SetFriction(0.8)
    contact_mat.SetRestitution(0.3)

    ground = chrono.ChBodyEasyBox(0.1,0.005,0.05,geom.rho,True,True,contact_mat)
    ground.SetPos(chrono.ChVectorD(lk[4][1,0],lk[4][1,1]-0.004,0))
    ground.SetBodyFixed(True)
    system.Add(ground)

    m_leg = 0
    links = []
    for i in range(len(lk)):
        if i == 7:
            tl = stiffness.tf(ct)
            wl = wf
        else:
            tl = motion.tr
            wl = motion.wr

        pos, rot, length = geom.pose(lk[i])

        m_leg += length*tl*wl*geom.rho

        if i == 4:
            link = chrono.ChBodyEasyBox(length,tl,wl,geom.rho,True,True,contact_mat)
        else:
            link = chrono.ChBodyEasyBox(length,tl,wl,geom.rho,True)
        link.SetPos(chrono.ChVectorD(*pos,0))
        link.SetRot(chrono.Q_from_AngZ(rot))
        system.Add(link)
        links.append(link)

    m_body = jump.cs['m']-m_leg
    w_body = (m_body/geom.rho)**(1/3)
    print('Leg mass',m_leg)
    print('Body mass',m_body)

    body = chrono.ChBodyEasyBox(w_body,w_body,w_body,geom.rho,True)
    body.SetPos(chrono.ChVectorD(0,0,0))
    system.Add(body)

    body_ground_link = chrono.ChLinkMateGeneric(True,False,True,True,True,True)
    body_ground_link.Initialize(body,ground,chrono.ChFrameD(chrono.ChVectorD(0,0,0)))
    system.Add(body_ground_link)

    body_damping_link = chrono.ChLinkTSDA()
    body_damping_link.Initialize(body,ground,False,chrono.ChVectorD(0,0,0),chrono.ChVectorD(0,0,0))
    body_damping_link.SetDampingCoefficient(0.2)
    system.AddLink(body_damping_link)

    body_leg_link = chrono.ChLinkMateGeneric(True,True,True,True,True,True)
    body_leg_link.Initialize(links[3],body,chrono.ChFrameD(chrono.ChVectorD(0,0,0)))
    system.Add(body_leg_link)


    joints = []
    springs = []
    springTorques.clear()
    con = [
        ([links[1]], [False]),
        ([links[2],links[4]], [False,True]),
        ([links[3]],[False]),
        ([links[0]],[False]),
        ([],[]),
        ([links[0],links[6]],[True,False]),
        ([links[7]],[False]),
        ([links[8]],[False]),
        ([links[3],links[5]],[False,False]),
    ]
    for i in range(len(links)):
        bodies = con[i][0]
        fixed = con[i][1]
        pos = lk[i][1,:]

        if i == 7:
            k = stiffness.prbm_k(stiffness.tf(ct),ls[1],wf)
            b = 0.0001
        else:
            k = 0
            b = 0

        for bd,f in zip(bodies,fixed):
            if f:
                joint = chrono.ChLinkMateGeneric(True,True,True,True,True,f)
                joint.Initialize(links[i],bd,chrono.ChFrameD(chrono.ChVectorD(*pos,0)))
                system.Add(joint)
                joints.append(joint)
                continue

            joint = chrono.ChLinkLockRevolute()
            joint.Initialize(links[i],bd,chrono.ChCoordsysD(chrono.ChVectorD(*pos,0)))
            system.Add(joint)
            joints.append(joint)

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

    # Crank bound
    joint = joints[4]
    joint_limit = joint.GetLimit_Rz()
    joint_limit.SetActive(True)
    joint_limit.SetMin(0)
    joint_limit.SetMax(np.pi)

    # Motor arm bound
    joint = joints[-2]
    joint_limit = joint.GetLimit_Rz()
    joint_limit.SetActive(True)
    joint_limit.SetMin(-jump.cs['t']/jump.cs['r'])
    joint_limit.SetMax(0)

    # Spring bound
    joint = joints[-1]
    joint_limit = joint.GetLimit_Rz()
    joint_limit.SetActive(True)
    joint_limit.SetMin(-jump.cs['t']/ks)
    joint_limit.SetMax(np.pi)

    motor = chrono.ChLinkMotorRotationTorque()
    motor.Initialize(links[8],links[3],chrono.ChFrameD(chrono.ChVectorD(0,0,0)))
    motorTorque = MotorTorqueDC(links[3],links[8])
    motor.SetTorqueFunction(motorTorque)
    system.Add(motor)

    application = chronoirr.ChIrrApp(system, 'leg', chronoirr.dimension2du(800, 600),chronoirr.VerticalDir_Y)
    application.AddTypicalSky()
    application.AddTypicalLights()
    y_offset = -0.04
    z_offset = -0.15
    application.AddTypicalCamera(chronoirr.vector3df(0, y_offset, z_offset),chronoirr.vector3df(0, y_offset, 0))
    application.AssetBindAll()
    application.AssetUpdateAll()

    application.SetTimestep(step)
    # application.SetVideoframeSaveInterval(int(1/step/2000))
    # application.SetVideoframeSave(True)

    datum = {
        't': [],
        'y': [],
        'dy': [],
        'grf': []
    }

    while application.GetDevice().run():
        application.BeginScene()
        application.DrawAll()

        datum['t'].append(system.GetChTime())
        datum['y'].append(body.GetPos().y)
        datum['dy'].append(body.GetPos_dt().y)
        datum['grf'].append(-ground.GetContactForce().y)

        # Draw axis for scale and orientation
        s = 0.1
        chronoirr.drawSegment(application.GetVideoDriver(),chrono.ChVectorD(0,0,0),chrono.ChVectorD(s,0,0),chronoirr.SColor(1,255,0,0))
        chronoirr.drawSegment(application.GetVideoDriver(),chrono.ChVectorD(0,0,0),chrono.ChVectorD(0,s,0),chronoirr.SColor(1,0,255,0))
        chronoirr.drawSegment(application.GetVideoDriver(),chrono.ChVectorD(0,0,0),chrono.ChVectorD(0,0,s),chronoirr.SColor(1,0,0,255))

        application.DoStep()
        application.EndScene()

        if system.GetChTime() > tfinal: application.GetDevice().closeDevice()

    return datum

for idx in [0,2,4]:
    datum = run(idx)
    plt.subplot(311)
    plt.plot(datum['t'],datum['y'])
    plt.subplot(312)
    plt.plot(datum['t'],datum['dy'])
    plt.subplot(313)
    plt.plot(datum['t'],datum['grf'])
plt.show()
