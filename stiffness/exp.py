import sys
sys.path.append('../utils')

import time
import matplotlib.pyplot as plt
import numpy as np
import math3d as m3d
import data
import ati
import urx
import plot

def to_movel(pose):
    return np.array(list(pose.pos)+list(pose.orient.rotation_vector))

# Base to foot of leg
Tbf = m3d.Transform()
Tbf.pos = m3d.Vector(0.000, -25.4*21/1000, 0.054)
Tbf.orient = m3d.Orientation.new_euler((0, 0, 0), encoding='XYZ')

# Tool pose wrt virtual rotation
Tft = m3d.Transform()
Tft.pos = m3d.Vector(0,0,0.13)
Tft.orient = m3d.Orientation.new_euler((np.pi, 0, np.pi/4), encoding='XYZ')

MOVEL_INIT = to_movel(Tbf*Tft)

if __name__ == '__main__':
    ur5 = urx.Robot("192.168.1.103")
    ur5.set_payload(0.1,(0,0,0))
    time.sleep(1)

    pose = ur5.get_pose()
    e = np.linalg.norm(to_movel(pose)[:3]-MOVEL_INIT[:3])
    print('Desired Pose', MOVEL_INIT)
    print('Actual Pose', to_movel(pose))
    print('Error', e)
    assert e < 0.001

    sensor = ati.init()

    time.sleep(2)

    t0 = time.time()
    dys = np.arange(0,-40.1,-1)/1000
    dys = np.concatenate(([0,0],dys))

    t = []
    z = []
    fz = []

    for dy in dys:
        Tffp = m3d.Transform()
        Tffp.pos = m3d.Vector(0,0,dy)
        Tffp.orient = m3d.Orientation.new_euler((0, 0, 0), encoding='XYZ')

        pose_d = to_movel(Tbf*Tffp*Tft)
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

    file_name = '../data/test.csv'
    data.write(
        file_name,
        ['t','z','fz'],
        [t,z,fz]
    )

    z,fz,zp,fzp,kp = read()
    plt.plot(z,fz,'.')
    plt.plot(zp,fzp)
    plt.show()
