import sys
sys.path.append('../utils')

import time
import numpy as np
import math3d as m3d
import data
import ati
import urx

RAD_PER_COUNT = 0.088/180*np.pi

# 0.000 .0125, .025, .0375, .050
# 25, 37.5, 50, 62.5, 75
TCP_Y_OFFSET = 0.000
POS_MID = 45/180*np.pi
STEP_NUM = 10 # signle side
STEP_SIZE = 20*RAD_PER_COUNT # 0.088 deg/count
POS_LIST = POS_MID+STEP_SIZE*np.concatenate((
    np.arange(0,STEP_NUM),
    np.arange(STEP_NUM,-STEP_NUM,-1),
    np.arange(-STEP_NUM,1),
))
t = m3d.Transform()
t.orient = m3d.Orientation.new_euler((np.pi, 0, POS_MID), encoding='XYZ')
L_INIT = np.array([-0.002, -0.38745, 0.080]+list(t.orient.rotation_vector))

def to_p(rot):
    return round((rot-POS_MID)/RAD_PER_COUNT+2048)

def limit_tool(rz):
    assert np.abs(rz) < POS_MID+STEP_NUM*STEP_SIZE, 'Tool rotation out of range'
    return

# Only z, absolute
def rotate_tool(rot):
    rot_c = ur5.get_pose().orient.to_euler('XYZ')[2]
    dr = rot-rot_c

    limit_tool(dr)
    t = m3d.Transform()
    t.orient.rotate_zb(dr)
    ur5.add_pose_tool(t,acc=0.1,vel=0.05,wait=False,command='movel')

if __name__ == '__main__':
    ur5 = urx.Robot("192.168.1.103")
    ur5.set_tcp((TCP_Y_OFFSET*np.cos(POS_MID),TCP_Y_OFFSET*np.sin(POS_MID),0,0,0,0))
    ur5.set_payload(0,(0,0,0))
    time.sleep(1)

    pose = ur5.get_pose()
    rot = pose.orient.to_euler('XYZ')

    e_pos = np.linalg.norm(L_INIT[:3]-np.array(list(pose.pos)))
    e_rotxy = np.linalg.norm(np.array([np.pi,0])-np.array(list(rot)[:2]))
    e_rotxy_i = np.linalg.norm(np.array([-np.pi,0])-np.array(list(rot)[:2]))
    assert e_pos < 0.001 and np.minimum(e_rotxy,e_rotxy_i) < 0.01, 'Move robot to L_INIT in pendant, e_pos {:.4f} e_rotxy {:.3f}'.format(e_pos,e_rotxy)
    print('Inital position OK')
    limit_tool(rot[2])
    print('Inital tool rotation OK')

    rotate_tool(POS_MID)
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

            rot = ur5.get_pose().orient.to_euler('XYZ')[2]
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
    ur5.set_tcp((0,0,0,0,0,0))
    ur5.close()

    file_name = '../data/test.csv'
    data.write(
        file_name,
        ['t','pd','p','tz','rotd','rot'],
        [ts,pds,ps,tzs,rotds,rots]
    )
