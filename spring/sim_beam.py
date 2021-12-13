import sys
sys.path.append('../utils')

import pychrono as chrono
import pychrono.fea as fea
import pychrono.irrlicht as chronoirr
import pychrono.pardisomkl as mkl
import numpy as np
import matplotlib.pyplot as plt
import process
import data

chrono.SetChronoDataPath('../chrono_data/')

E = 18.6e9
mu = 0.3
rho = 1820
dl = 0.001
lmax = 0.09

step = 5e-4
tfinal = 2

def run(tv,wv,lv,type='euler'):
    t = tv*2.54e-5
    w = wv/1000
    l = lv/1000

    system = chrono.ChSystemNSC()
    system.Set_G_acc(chrono.ChVectorD(0,0,0))

    ground = chrono.ChBodyEasyCylinder(t,w,1,True)
    ground.SetPos(chrono.ChVectorD(0,0,0))
    ground.SetBodyFixed(True)
    system.Add(ground)

    pin = chrono.ChBodyEasyCylinder(t,w,1,True)
    pin.SetPos(chrono.ChVectorD(l,0,0))
    system.Add(pin)

    mesh = fea.ChMesh();

    if type == 'iga':
        inertia = fea.ChInertiaCosseratSimple()
        inertia.SetAsRectangularSection(w,t,rho)

        elasticity = fea.ChElasticityCosseratSimple()
        elasticity.SetYoungModulus(E)
        elasticity.SetGwithPoissonRatio(mu)
        elasticity.SetAsRectangularSection(w,t)

        section = fea.ChBeamSectionCosserat(inertia, elasticity)
        section.SetDrawThickness(w,t)

        builder = fea.ChBuilderBeamIGA()
        builder.BuildBeam(mesh,section,int(l/dl),chrono.ChVectorD(0,0,0),chrono.ChVectorD(l,0,0),chrono.ChVectorD(0,1,0),1)
    else:
        section = fea.ChBeamSectionEulerSimple()
        section.SetYoungModulus(E)
        section.SetGwithPoissonRatio(mu)
        section.SetDensity(rho)
        section.SetAsRectangularSection(w,t)
        # section.SetBeamRaleyghDamping(0.1)

        builder = fea.ChBuilderBeamEuler()
        builder.BuildBeam(mesh,section,int(l/dl),chrono.ChVectorD(0,0,0),chrono.ChVectorD(l,0,0),chrono.ChVectorD(0,1,0))

    builder.GetLastBeamNodes()[0].SetFixed(True)

    system.Add(mesh)

    link = chrono.ChLinkMateGeneric(False,True,True,False,False,False)
    link.Initialize(
        pin,
        builder.GetLastBeamNodes()[int(l/dl)],
        chrono.ChFrameD(chrono.ChVectorD(l,0,0))
    )
    system.Add(link)

    motor = chrono.ChLinkMotorRotationAngle()
    motor.Initialize(
        ground,
        pin,
        chrono.ChFrameD(chrono.ChVectorD(0,0,0),chrono.Q_from_AngX(chrono.CH_C_PI/2))
    )
    motor.SetAngleFunction(chrono.ChFunction_Sine(0,1/tfinal,30/180*chrono.CH_C_PI))
    system.Add(motor)

    solver = mkl.ChSolverPardisoMKL()
    system.SetSolver(solver)

    v_surf = fea.ChVisualizationFEAmesh(mesh)
    v_surf.SetFEMdataType(fea.ChVisualizationFEAmesh.E_PLOT_SURFACE)
    mesh.AddAsset(v_surf)

    application = chronoirr.ChIrrApp(system, "Spring", chronoirr.dimension2du(800, 600),chronoirr.VerticalDir_Y)
    application.AddTypicalSky()
    application.AddTypicalLights()
    application.AddTypicalCamera(chronoirr.vector3df(0,0.2,0),chronoirr.vector3df(0,0,0))
    application.AssetBindAll()
    application.AssetUpdateAll()
    application.SetTimestep(step)

    T = []
    P = []
    TZ = []
    ROT = []
    while application.GetDevice().run():
        T.append(system.GetChTime())
        TZ.append(-motor.GetMotorTorque())

        rot = motor.GetMotorRot()
        p = int(rot/chrono.CH_C_PI*180/process.DEG_PER_COUNT)+process.POS_MID
        P.append(p)
        ROT.append(rot)

        application.BeginScene()
        application.DrawAll()

        # Draw axis for scale and orientation
        l = 0.1
        chronoirr.drawSegment(application.GetVideoDriver(),chrono.ChVectorD(0,0,0),chrono.ChVectorD(l,0,0),chronoirr.SColor(1,255,0,0))
        chronoirr.drawSegment(application.GetVideoDriver(),chrono.ChVectorD(0,0,0),chrono.ChVectorD(0,l,0),chronoirr.SColor(1,0,255,0))
        chronoirr.drawSegment(application.GetVideoDriver(),chrono.ChVectorD(0,0,0),chrono.ChVectorD(0,0,l),chronoirr.SColor(1,0,0,255))

        application.DoStep()
        application.EndScene()

        if system.GetChTime() > tfinal: # in system seconds
              application.GetDevice().closeDevice()

    file_name = '../data/{:d}mil_{:d}mm_{:d}mm_sim_beam.csv'.format(tv,lv,wv)
    data.write(
        file_name,
        ['t','p','tz','rot'],
        [T,P,TZ,ROT]
    )

for t in [30]:
    for w in [10,20,30]:
        for l in [25,50,75]:
            run(t,w,l)
