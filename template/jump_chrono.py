import pychrono as chrono
import pychrono.irrlicht as chronoirr
import numpy as np
import matplotlib.pyplot as plt

chrono.SetChronoDataPath('../chrono_data/')

pi = np.pi
rho = 1000
step = 5e-5
tfinal = 1

mb = 0.03
wb = (mb/rho)**(1/3)
lb = 0.05

ml = 0.005
wl = (ml/rho)**(1/3)
ll = 0.05

rf = 0.005

k = 200
a = 1
el = 0.1

tau = 0.215
v = 383/60*2*pi
em = pi
r = 0.06
b = tau/r/(v*r)

system = chrono.ChSystemNSC()
system.Set_G_acc(chrono.ChVectorD(0,-9.81,0))

chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.001)
chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.001)

contact_mat = chrono.ChMaterialSurfaceNSC()
contact_mat.SetFriction(0.3)
contact_mat.SetRestitution(0.3)

ground = chrono.ChBodyEasyBox(1.0,0.1,0.1,rho,True,True,contact_mat)
ground.SetPos(chrono.ChVectorD(0,-0.05,0))
ground.SetRot(chrono.Q_from_AngZ(0))
ground.SetBodyFixed(True)
ground.GetCollisionModel().SetFamily(0)
system.Add(ground)

body = chrono.ChBodyEasyBox(wb,wb,wb,rho,True)
body.SetPos(chrono.ChVectorD(0,ll+lb,0))
# body.SetBodyFixed(True)
system.Add(body)

leg = chrono.ChBodyEasyBox(wl,wl,wl,rho,True)
leg.SetPos(chrono.ChVectorD(0,ll,0))
# leg.SetBodyFixed(True)
system.Add(leg)

foot = chrono.ChBodyEasySphere(rf,rho,True,True,contact_mat)
foot.SetPos(chrono.ChVectorD(0,rf,0))
system.Add(foot)

# Links
# joint_body_leg = chrono.ChLinkMateGeneric(True,True,True,True,True,True)
# joint_body_leg.Initialize(body,leg,chrono.ChFrameD(chrono.ChVectorD(0,ll+lb,0)))
# system.Add(joint_body_leg)

joint_body_ground = chrono.ChLinkMateGeneric(True,False,True,True,True,True)
joint_body_ground.Initialize(body,ground,chrono.ChFrameD(chrono.ChVectorD(0,0,0)))
system.Add(joint_body_ground)

joint_leg_ground = chrono.ChLinkMateGeneric(True,False,True,True,True,True)
joint_leg_ground.Initialize(leg,ground,chrono.ChFrameD(chrono.ChVectorD(0,0,0)))
system.Add(joint_leg_ground)

joint_foot_ground = chrono.ChLinkMateGeneric(True,False,True,True,True,True)
joint_foot_ground.Initialize(foot,ground,chrono.ChFrameD(chrono.ChVectorD(0,0,0)))
system.Add(joint_foot_ground)

damper = chrono.ChLinkTSDA()
damper.SetSpringCoefficient(0)
damper.SetDampingCoefficient(b)
damper.Initialize(
    body,leg,False,
    chrono.ChVectorD(0,ll+lb,0),chrono.ChVectorD(0,ll,0),
    True
)
system.AddLink(damper)

class SpringForce(chrono.ForceFunctor):
    def __init__(self):
        super().__init__()

    def __call__(self,time,rest_length,length,vel,link):
        l = length-rest_length
        force = -np.sign(l)*k*el*np.power(np.abs(l/el),a)
        return force

spring = chrono.ChLinkTSDA()
spring.Initialize(
    leg,foot,False,
    chrono.ChVectorD(0,ll,0),chrono.ChVectorD(0,rf,0),
    True
)
springForce = SpringForce()
spring.RegisterForceFunctor(springForce)
system.AddLink(spring)

# Motor
class MotorTorque(chrono.ChFunction) :
    def __init__(self, body, leg):
        super().__init__()
        self.body = body
        self.leg = leg

    def Clone(self):
      return deepcopy(self)

    def Get_y(self, x):
        dy = self.body.GetPos().y-self.leg.GetPos().y
        dyd = em*r+lb
        return np.maximum(0,dy-dyd)*10000-tau/r

motor = chrono.ChLinkMotorLinearForce()
motor.Initialize(leg,body,chrono.ChFrameD(chrono.ChVectorD(0,ll+lb,0),chrono.Q_from_AngZ(pi/2)))
motorTorque = MotorTorque(body, leg)
motor.SetForceFunction(motorTorque)
system.Add(motor)

# Visuals
visual_spring = chrono.ChPointPointSpring(0.001, 100, 20)
color_red = chrono.ChColorAsset()
color_red.SetColor(chrono.ChColor(1.0, 0, 0))
color_green = chrono.ChColorAsset()
color_green.SetColor(chrono.ChColor(0, 1.0, 0))

spring.AddAsset(color_red)
spring.AddAsset(visual_spring)

application = chronoirr.ChIrrApp(system, "Jump", chronoirr.dimension2du(1024, 768),chronoirr.VerticalDir_Y)
application.AddTypicalSky()
application.AddTypicalLights()
y_offset = 0.1
z_offset = -0.2
application.AddTypicalCamera(chronoirr.vector3df(0, y_offset, z_offset),chronoirr.vector3df(0, y_offset, 0))
application.AssetBindAll()
application.AssetUpdateAll()

application.SetTimestep(step)
# application.SetVideoframeSaveInterval(int(1/step/2000))
# application.SetVideoframeSave(True)

ts = []
ybs = []
dybs = []
yls = []
dyls = []

while application.GetDevice().run():
    t = system.GetChTime()
    yb = body.GetPos().y
    dyb = body.GetPos_dt().y
    yl = leg.GetPos().y
    dyl = leg.GetPos_dt().y
    ts.append(t)
    ybs.append(yb)
    dybs.append(dyb)
    yls.append(yl)
    dyls.append(dyl)

    application.BeginScene()
    application.DrawAll()

    # Draw axis for scale and orientation
    l = 0.1
    chronoirr.drawSegment(application.GetVideoDriver(),chrono.ChVectorD(0,0,0),chrono.ChVectorD(l,0,0),chronoirr.SColor(1,255,0,0))
    chronoirr.drawSegment(application.GetVideoDriver(),chrono.ChVectorD(0,0,0),chrono.ChVectorD(0,l,0),chronoirr.SColor(1,0,255,0))
    chronoirr.drawSegment(application.GetVideoDriver(),chrono.ChVectorD(0,0,0),chrono.ChVectorD(0,0,l),chronoirr.SColor(1,0,0,255))

    application.DoStep()
    application.EndScene()

    if yl-ll > 1e-6 :
         application.GetDevice().closeDevice()

    # if dyb+1e-6 < 0:
    #      application.GetDevice().closeDevice()

    if system.GetChTime() > tfinal: # in system seconds
          application.GetDevice().closeDevice()

plt.figure()
plt.subplot(221)
plt.plot(ts,ybs)
plt.subplot(222)
plt.plot(ts,yls)
plt.subplot(223)
plt.plot(ts,dybs)
plt.subplot(224)
plt.plot(ts,dyls)
plt.show()
