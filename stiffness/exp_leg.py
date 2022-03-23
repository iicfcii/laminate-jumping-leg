import time
import matplotlib.pyplot as plt
import numpy as np
import math3d as m3d
import urx
from utils import data
from utils import ati

def to_movel(pose):
    return np.array(list(pose.pos)+list(pose.orient.rotation_vector))

def str_movel(pose):
    return '{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(
        *pose[:3]*1000,
        *pose[3:],
    )

# Base to foot of leg
Tbc = m3d.Transform()
Tbc.pos = m3d.Vector(-4/1000, (-25.4*21-1-30)/1000, 0.168)
Tbc.orient = m3d.Orientation.new_euler((0, 0, -np.pi/2), encoding='XYZ')

# Tool pose wrt virtual rotation
Tct = m3d.Transform()
Tct.pos = m3d.Vector(-75/1000,0,0)
Tct.orient = m3d.Orientation.new_euler((np.pi, 0, np.pi/4), encoding='XYZ')

MOVEL_INIT = to_movel(Tbc*Tct)
Z_MAX = 0.06

if __name__ == '__main__':
    ur5 = urx.Robot("192.168.1.103")
    ur5.set_payload(0.1,(0,0,0))
    time.sleep(1)

    pose = ur5.get_pose()
    e = np.linalg.norm(to_movel(pose)[:3]-MOVEL_INIT[:3])
    print('Desired Pose', str_movel(MOVEL_INIT))
    print('Actual Pose', str_movel(to_movel(pose)))
    print('Error', e)
    assert e < 0.001 # Only position

    sensor = ati.init()

    time.sleep(2)

    t0 = time.time()
    dzs = np.arange(0,-Z_MAX,-0.0005)
    dzs = np.concatenate(([0,0],dzs))

    t = []
    z = []
    fz = []

    for dz in dzs:
        Tccp = m3d.Transform()
        Tccp.pos = m3d.Vector(0,0,dz)

        pose_d = to_movel(Tbc*Tccp*Tct)
        ur5.movel(pose_d,acc=0.1,vel=0.05,wait=False)

        t0p = time.time()
        while time.time()-t0p < 1: # Wait 1 second
            pose_c = ur5.get_pose()

            ati.single_read(sensor)
            f = ati.recv(sensor)

            t.append(time.time()-t0)
            z.append(pose_c.pos[2])
            fz.append(f[2])

    ur5.movel(MOVEL_INIT,acc=0.1,vel=0.05,wait=False)

    ati.stop(sensor)
    ur5.close()

    file_name = './data/test.csv'
    data.write(
        file_name,
        ['t','z','fz'],
        [t,z,fz]
    )
