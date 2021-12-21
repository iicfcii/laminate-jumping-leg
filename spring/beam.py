import sys
sys.path.append('../utils')

import pychrono as chrono
import pychrono.fea as fea
import pychrono.irrlicht as chronoirr
import pychrono.pardisomkl as mkl
import numpy as np
import matplotlib.pyplot as plt
import data
import process

chrono.SetChronoDataPath('../chrono_data/')

E = 18.6e9*1.1
mu = 0.3
rho = 1820
dl = 0.005
lmax = 0.09

step = 5e-4
tfinal = 0.1

def sim(rot,tz,tmil,lmm,wmm):
    t = tmil*2.54e-5
    w = wmm/1000
    l = lmm/1000

    system = chrono.ChSystemNSC()
    system.Set_G_acc(chrono.ChVectorD(0,0,0))

    mesh = fea.ChMesh();

    section = fea.ChBeamSectionEulerSimple()
    section.SetYoungModulus(E)
    section.SetGwithPoissonRatio(mu)
    section.SetDensity(rho)
    section.SetAsRectangularSection(w,t)
    # section.SetBeamRaleyghDamping(0.1)

    builder = fea.ChBuilderBeamEuler()
    builder.BuildBeam(mesh,section,int(lmax/dl),chrono.ChVectorD(0,0,0),chrono.ChVectorD(lmax,0,0),chrono.ChVectorD(0,1,0))

    builder.GetLastBeamNodes()[0].SetFixed(True)
    builder.GetLastBeamNodes()[int(l/dl)].SetForce(chrono.ChVectorD(tz/l*np.sin(rot),0,tz/l*np.cos(rot))),

    system.Add(mesh)

    solver = mkl.ChSolverPardisoMKL()
    system.SetSolver(solver)

    v_surf = fea.ChVisualizationFEAmesh(mesh)
    v_surf.SetFEMdataType(fea.ChVisualizationFEAmesh.E_PLOT_SURFACE)
    mesh.AddAsset(v_surf)

    system.SetChTime(0)
    while True:
        system.DoStepDynamics(step)
        if system.GetChTime() > tfinal: break

    pos = builder.GetLastBeamNodes()[int(l/dl)].GetPos()
    return pos.x, pos.z

if __name__ == '__main__':
    for tmil in [16.5,32.5]:
        for wmm in [10,20,30]:
            for lmm in [25,50,75]:
                rot, tz = process.sample(tmil,lmm,wmm,'1',np.pi/4 if tmil==32.5 else 0)

                x = []
                y = []
                for r,t in zip(rot,tz):
                    pos = sim(r,t,tmil,lmm,wmm)
                    x.append(pos[0])
                    y.append(pos[1])

                file_name = '../data/{:d}mil_{:d}mm_{:d}mm_beam.csv'.format(int(tmil/5)*5,int(lmm),wmm)
                data.write(
                    file_name,
                    ['x','y','rot','tz'],
                    [x,y,rot,tz]
                )

                print(tmil,lmm,wmm,'done')
