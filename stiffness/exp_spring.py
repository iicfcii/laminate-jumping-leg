import sys
sys.path.append('../utils')

import time
import matplotlib.pyplot as plt
import numpy as np
import math3d as m3d
import data
import ati
import urx
import plot_spring

def to_movel(pose):
    return np.array(list(pose.pos)+list(pose.orient.rotation_vector))

def str_movel(pose):
    return '{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}'.format(
        *pose[:3]*1000,
        *pose[3:],
    )

# Base to foot of leg
Tbc = m3d.Transform()
Tbc.pos = m3d.Vector(-4/1000, (-25.4*22-1)/1000, 0.070)
Tbc.orient = m3d.Orientation.new_euler((0, 0, -np.pi/2+30/180*np.pi), encoding='XYZ')

# Tool pose wrt virtual rotation
Tct = m3d.Transform()
Tct.pos = m3d.Vector(-70/1000,0,0)
Tct.orient = m3d.Orientation.new_euler((np.pi, 0, np.pi/4), encoding='XYZ')

MOVEL_INIT = to_movel(Tbc*Tct)

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
    drs = np.arange(0,0.9,0.02)
    drs = np.concatenate(([0,0],drs))

    t = []
    rz = []
    tz = []

    for dr in drs:
        Tccp = m3d.Transform()
        Tccp.pos = m3d.Vector(0,0,0)
        Tccp.orient = m3d.Orientation.new_euler((0,0,dr), encoding='XYZ')

        pose_d = to_movel(Tbc*Tccp*Tct)
        ur5.movel(pose_d,acc=0.1,vel=0.05,wait=False)

        t0p = time.time()
        while time.time()-t0p < 1: # Wait 1 second
            pose_c = ur5.get_pose()

            ati.single_read(sensor)
            f = ati.recv(sensor)

            t.append(time.time()-t0)
            rz.append(pose_c.orient.to_euler('XYZ')[2])
            tz.append(f[5])

    ur5.movel(MOVEL_INIT,acc=0.1,vel=0.05,wait=False)

    ati.stop(sensor)
    ur5.close()

    file_name = '../data/test.csv'
    data.write(
        file_name,
        ['t','rz','tz'],
        [t,rz,tz]
    )

    rz,tz,kp = plot_spring.read()
    print(kp)
    rzp = np.linspace(0,0.8,100)
    tzp = kp*rzp
    plt.plot(rz,tz,'.')
    plt.plot(rzp,tzp)
    plt.show()
