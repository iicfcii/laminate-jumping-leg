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

pi = np.pi

E = 18.6e9
mu = 0.3
rho = 1820
dl = 0.001
lmax = 0.09

step = 2e-4
tfinal = 2

def cast_node(nb):
    feaNB = fea.CastToChNodeFEAbase(nb)
    nodeFead = fea.CastToChNodeFEAxyzD(feaNB)
    return nodeFead

def run(tv,wv,lv,vis=True):
    t = tv*2.54e-5
    w = wv/1000
    l = lv/1000

    system = chrono.ChSystemSMC()
    system.Set_G_acc(chrono.ChVectorD(0,0,0))

    ground = chrono.ChBodyEasyCylinder(t,w,1,True)
    ground.SetPos(chrono.ChVectorD(0,0,0))
    ground.SetBodyFixed(True)
    system.Add(ground)

    pin = chrono.ChBodyEasyCylinder(t,w,1,True)
    pin.SetPos(chrono.ChVectorD(l,0,0))
    system.Add(pin)

    mesh = fea.ChMesh()

    num_div_x = int(l/dl)
    # num_div_y = int(w/dl)
    num_div_y = 1
    num_node_x = num_div_x+1
    num_node_y = num_div_y+1

    num_elements = num_div_x*num_div_y
    num_nodes = num_node_x*num_node_y

    dx = l/num_div_x
    dy = w/num_div_y
    dz = t

    for j in range(num_node_y):
        for i in range(num_node_x):
            # Position of node
            x = i*dx
            y = j*dy-w/2
            z = 0

            dirX = 0
            dirY = 0
            dirZ = -1

            # If nodes added to element in CCW then -y
            node = fea.ChNodeFEAxyzD(
                chrono.ChVectorD(x,y,z),
                chrono.ChVectorD(dirX,dirY,dirZ),
            )
            node.SetMass(0)

            if i == 1: node.SetFixed(True)

            if i == int(l/dl):
                link = fea.ChLinkPointFrameGeneric(False,True,True)
                link.Initialize(node,pin)
                system.Add(link)

            mesh.AddNode(node)

    mat = fea.ChMaterialShellANCF(rho, E, mu)
    for j in range(num_div_y):
        for i in range(num_div_x):
            nodeA = i+j*num_node_x
            nodeB = i+j*num_node_x+1
            nodeC = i+(j+1)*num_node_x+1
            nodeD = i+(j+1)*num_node_x

            element = fea.ChElementShellANCF_3423()
            element.SetNodes(
                cast_node(mesh.GetNode(nodeA)),
                cast_node(mesh.GetNode(nodeB)),
                cast_node(mesh.GetNode(nodeC)),
                cast_node(mesh.GetNode(nodeD))
            )
            element.SetDimensions(dx, dy)
            element.AddLayer(dz, 0*chrono.CH_C_DEG_TO_RAD, mat)
            element.SetAlphaDamp(0.001)

            mesh.AddElement(element)

    system.Add(mesh)

    motor = chrono.ChLinkMotorRotationAngle()
    motor.Initialize(
        ground,
        pin,
        chrono.ChFrameD(chrono.ChVectorD(0,0,0),chrono.Q_from_AngX(pi/2))
    )
    motor.SetAngleFunction(chrono.ChFunction_Sine(0,1/tfinal,30/180*pi))
    # motor.SetSpindleConstraint(False,False,False,False,False)
    system.Add(motor)

    # Visuals
    vbeam = fea.ChVisualizationFEAmesh(mesh)
    vbeam.SetFEMdataType(fea.ChVisualizationFEAmesh.E_PLOT_SURFACE)
    mesh.AddAsset(vbeam)

    mkl_solver = mkl.ChSolverPardisoMKL()
    system.SetSolver(mkl_solver)
    # solver = chrono.ChSolverSparseQR()
    # solver.UseSparsityPatternLearner(True)
    # solver.LockSparsityPattern(True)
    # solver.SetVerbose(False)
    # system.SetSolver(solver)

    application = chronoirr.ChIrrApp(system, "Spring Shell", chronoirr.dimension2du(800, 600), chronoirr.VerticalDir_Y)
    application.AddTypicalSky()
    application.AddTypicalLights()
    application.AddTypicalCamera(chronoirr.vector3df(0.1,0.1,0.1),chronoirr.vector3df(0,0,0))
    application.AssetBindAll()
    application.AssetUpdateAll()

    application.SetTimestep(step)

    T = []
    P = []
    TZ = []
    ROT = []
    while application.GetDevice().run():
        T.append(system.GetChTime())
        TZ.append(motor.GetMotorTorque())

        rot = motor.GetMotorRot()
        p = int(rot/pi*180/process.DEG_PER_COUNT)+process.POS_MID
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

        if system.GetChTime() > tfinal: application.GetDevice().closeDevice()

    file_name = '../data/{:d}mil_{:d}mm_{:d}mm_sim_shell.csv'.format(tv,lv,wv)
    data.write(
        file_name,
        ['t','p','tz','rot'],
        [T,P,TZ,ROT]
    )

for t in [30]:
    for w in [10,20]:
        for l in [25,50,75]:
            run(t,w,l)
