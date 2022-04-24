import pychrono as chrono
import pychrono.irrlicht as chronoirr
import matplotlib.pyplot as plt
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

xm = design.xm
ks = design.springs[3]['k']
xs = design.springs[3]['x']
tfinal = 1
step = 1e-4

print(xm)
print(xs)

ang = xm[0]
l = xm[1:6]
c = xm[6]
ls = xs[:4]
cm = xs[4]
ct = xs[5]
wf = xs[6]
lk = geom.leg_spring(ang,l,c,ls,cm,motion.tr)

system = chrono.ChSystemNSC()
system.Set_G_acc(chrono.ChVectorD(0,-9.81*0,0))

chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.001)
chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.001)
contact_mat = chrono.ChMaterialSurfaceNSC()
contact_mat.SetFriction(0.8)
contact_mat.SetRestitution(0.3)

ground = chrono.ChBodyEasyBox(0.1,0.005,0.05,geom.rho,True,True,contact_mat)
ground.SetPos(chrono.ChVectorD(lk[4][1,0],lk[4][1,1]-0.004,0))
ground.SetBodyFixed(True)
system.Add(ground)

links = []
for i in range(len(lk)):
    tl = motion.tr
    wl = motion.wr

    pos, rot, length = geom.pose(lk[i])

    if i == 4:
        link = chrono.ChBodyEasyBox(length,tl,wl,geom.rho,True,True,contact_mat)
    else:
        link = chrono.ChBodyEasyBox(length,tl,wl,geom.rho,True)
    link.SetPos(chrono.ChVectorD(*pos,0))
    link.SetRot(chrono.Q_from_AngZ(rot))
    system.Add(link)
    links.append(link)
links[3].SetBodyFixed(True)

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
    ([links[5]],[False]),
]
for i in range(len(links)):
    bodies = con[i][0]
    fixed = con[i][1]

    pos = lk[i][1,:]

    if i == 7:
        k = stiffness.prbm_k(stiffness.tf(ct),ls[1],wf)
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
motor.Initialize(links[8],links[3],chrono.ChFrameD(chrono.ChVectorD(0,0,0)))
motor.SetAngleFunction(chrono.ChFunction_Ramp(0,-(jump.cs['t']/ks)/tfinal))
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
    'rz': [],
    'fx': [],
    'fy': [],
}

while application.GetDevice().run():
    application.BeginScene()
    application.DrawAll()

    datum['t'].append(system.GetChTime())
    datum['rz'].append(-motor.GetMotorRot())
    datum['fx'].append(-ground.GetContactForce().x)
    datum['fy'].append(-ground.GetContactForce().y)

    # Draw axis for scale and orientation
    s = 0.1
    chronoirr.drawSegment(application.GetVideoDriver(),chrono.ChVectorD(0,0,0),chrono.ChVectorD(s,0,0),chronoirr.SColor(1,255,0,0))
    chronoirr.drawSegment(application.GetVideoDriver(),chrono.ChVectorD(0,0,0),chrono.ChVectorD(0,s,0),chronoirr.SColor(1,0,255,0))
    chronoirr.drawSegment(application.GetVideoDriver(),chrono.ChVectorD(0,0,0),chrono.ChVectorD(0,0,s),chronoirr.SColor(1,0,0,255))

    application.DoStep()
    application.EndScene()

    if system.GetChTime() > tfinal: application.GetDevice().closeDevice()

plt.plot(datum['rz'],datum['fy'])
# plt.plot(datum['rz'],datum['fx'])
plt.show()
