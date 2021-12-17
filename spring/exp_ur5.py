import sys
sys.path.append('../utils')

import time
import numpy as np
import math3d as m3d
import data
import ati
import urx

RAD_PER_COUNT = 0.088/180*np.pi

POS_MID = 0/180*np.pi
STEP_NUM = 10 # signle side
STEP_SIZE = 20/STEP_NUM/180*np.pi # 0.088 deg/count
POS_LIST = STEP_SIZE*np.concatenate((
    np.arange(0,STEP_NUM),
    np.arange(STEP_NUM,-STEP_NUM,-1),
    np.arange(-STEP_NUM,1),
))

VIRTUAL_CENTER_OFFSET = -0.0
TOOL_ROT_OFFSET = np.pi/4

# Virtual rotation pose wrt base
Tbv = m3d.Transform()
Tbv.pos = m3d.Vector(-0.0025, -0.38645, 0.080)
Tbv.orient = m3d.Orientation.new_euler((np.pi, 0, 0), encoding='XYZ')

# Tool pose wrt virtual rotation
Tvt = m3d.Transform()
Tvt.pos = m3d.Vector(0,VIRTUAL_CENTER_OFFSET,0)
Tvt.orient = m3d.Orientation.new_euler((0, 0, TOOL_ROT_OFFSET), encoding='XYZ')

def to_l(pose):
    return np.array(list(pose.pos)+list(pose.orient.rotation_vector))

def to_p(rot):
    return round((rot-POS_MID)/RAD_PER_COUNT+2048)

def limit_tool(rz):
    assert np.abs(rz) < POS_MID+STEP_NUM*STEP_SIZE, 'Tool rotation out of range'
    return

def rotate_tool(rot): # Only z, absolute
    Tvv = m3d.Transform()
    Tvv.orient.rotate_zb(rot)
    Tbt = Tbv*Tvv*Tvt

    ur5.movel(to_l(Tbt),acc=0.1,vel=0.05,wait=False)

# Tool pose wrt base
Tbt = Tbv*Tvt
L_INIT = to_l(Tbt)
print('Desired initial pose',list(L_INIT))

if __name__ == '__main__':
    ur5 = urx.Robot("192.168.1.103")
    ur5.set_payload(0.17,(0,-0.002,0.013))
    time.sleep(1)

    pose = ur5.get_pose()
    pos = np.array(list(pose.pos))
    rot = np.array(list(pose.orient.to_euler('XYZ')))
    print('Actual initial pose',list(pos)+list(pose.orient.rotation_vector))

    e_pos = np.linalg.norm(L_INIT[:3]-pos)
    e_rotxy = np.linalg.norm(np.array([np.pi,0])-rot[:2])
    e_rotxy_i = np.linalg.norm(np.array([-np.pi,0])-rot[:2])
    assert (
        e_pos < 0.001 and
        np.minimum(e_rotxy,e_rotxy_i) < 0.01
    ), \
    'Please move robot to L_INIT manually, e_pos {:.4f} e_rotxy {:.3f}'.format(e_pos,e_rotxy)
    print('Inital position OK')

    rotate_tool(0)
    time.sleep(2)
    print('Homed')

    sensor = ati.init()
    ati.set_bias(sensor)
    print('Force bias set')
    time.sleep(1)

    ts = []
    pds = []
    ps = []
    tzs = []
    rotds = []
    rots = []

    print('Experiment started')
    t0 = time.time()
    for rotd in POS_LIST:
        pd = to_p(rotd)
        print('pd',pd,'rotd',rotd)
        tp0 = time.time()
        pos_set = False
        while time.time()-tp0 < 1: # Wait 1 second
            if not pos_set:
                rotate_tool(rotd)
                pos_set = True

            rot = (Tbv.inverse*Tbt.inverse*(ur5.get_pose())).orient.to_euler('XYZ')[2]

            p = to_p(rot)

            ati.single_read(sensor)
            f = ati.recv(sensor)

            ts.append(time.time()-t0)
            pds.append(pd)
            ps.append(p)
            tzs.append(f[5])
            rotds.append(rotd)
            rots.append(rot)
            print('t {:.3f} rotd {:.3f} rot {:.3f} pd {:d} p {:d} tz {:.3f}'.format(time.time()-t0,rotd,rot,pd,p,f[5]))

    ati.stop(sensor)
    ur5.close()

    file_name = '../data/test.csv'
    data.write(
        file_name,
        ['t','pd','p','tz','rotd','rot'],
        [ts,pds,ps,tzs,rotds,rots]
    )
