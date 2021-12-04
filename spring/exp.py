import sys
sys.path.append('../utils')

import time
import numpy as np
from dynamixel_sdk import *
import ati
import data

PORT_NAME = 'COM9'
PROTOCOL_VERSION = 2.0
BAUDRATE = 57600
SERVO_ID = 1

ADDR_TORQUE_ENABLE = 64
LEN_TORQUE_ENABLE = 1
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4

POS_MID = 2068 # ~180 deg
STEP_NUM = 20 # signle side
STEP_SIZE = 20 # 0.088 deg/count
POS_LIST = POS_MID+STEP_SIZE*np.concatenate((
    np.arange(0,STEP_NUM),
    np.arange(STEP_NUM,-STEP_NUM,-1),
    np.arange(-STEP_NUM,1),
))

def limit_p(p):
    r = STEP_SIZE*STEP_NUM+50
    p_start = POS_MID-r
    p_end = POS_MID+r
    assert p > p_start and p < p_end, 'Position should be between {} and {} but is {}'.format(p_start,p_end,p)

if __name__ == '__main__':
    portH = PortHandler(PORT_NAME)
    portH.setBaudRate(BAUDRATE)
    packetH = PacketHandler(PROTOCOL_VERSION)

    # Disable torque and check current position
    # To avoid potential damage to test setup, especially force sensor
    groupSW = GroupSyncWrite(portH, packetH, ADDR_TORQUE_ENABLE, LEN_TORQUE_ENABLE)
    groupSW.addParam(SERVO_ID,int(0).to_bytes(LEN_TORQUE_ENABLE, 'little'))
    groupSW.txPacket()

    groupSR = GroupSyncRead(portH, packetH, ADDR_PRESENT_POSITION,LEN_PRESENT_POSITION)
    groupSR.addParam(SERVO_ID)
    groupSR.txRxPacket()
    current_pos = groupSR.getData(SERVO_ID,ADDR_PRESENT_POSITION,LEN_PRESENT_POSITION)

    limit_p(current_pos)
    print('Inital Position OK')

    # Enable torque and move to mid position
    groupSW = GroupSyncWrite(portH, packetH, ADDR_TORQUE_ENABLE, LEN_TORQUE_ENABLE)
    groupSW.addParam(SERVO_ID,int(1).to_bytes(LEN_TORQUE_ENABLE, 'little'))
    groupSW.txPacket()

    groupSW = GroupSyncWrite(portH, packetH, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
    groupSW.addParam(SERVO_ID,int(POS_MID).to_bytes(LEN_GOAL_POSITION, 'little'))
    groupSW.txPacket()

    print('Servo homed')
    time.sleep(3)

    sensor = ati.init()
    ati.set_bias(sensor)
    print('Force bias set')
    time.sleep(1)

    ts = []
    pds = []
    ps = []
    tzs = []

    print('Experiment started')
    t0 = time.time()
    for pd in POS_LIST:
        print('pd', pd)
        tp0 = time.time()
        pos_set = False
        while time.time()-tp0 < 1:
            if not pos_set:
                limit_p(pd)
                groupSW = GroupSyncWrite(portH, packetH, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
                groupSW.addParam(SERVO_ID,int(pd).to_bytes(LEN_GOAL_POSITION, 'little'))
                groupSW.txPacket()
                pos_set = True

            groupSR = GroupSyncRead(portH, packetH, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            groupSR.addParam(SERVO_ID)
            groupSR.txRxPacket()
            p = groupSR.getData(SERVO_ID,ADDR_PRESENT_POSITION,LEN_PRESENT_POSITION)

            ati.single_read(sensor)
            f = ati.recv(sensor)

            ts.append(time.time()-t0)
            pds.append(pd)
            ps.append(p)
            tzs.append(f[5])
            # print('t {:.3f} pd {} p {} tz {:.3f}'.format(time.time()-t0,pd,p,f[5]))

    # Disable torque
    groupSW = GroupSyncWrite(portH, packetH, ADDR_TORQUE_ENABLE, LEN_TORQUE_ENABLE)
    groupSW.addParam(SERVO_ID,int(0).to_bytes(LEN_TORQUE_ENABLE, 'little'))
    groupSW.txPacket()

    ati.stop(sensor)

    file_name = '../data/test.csv'
    data.write(
        file_name,
        ['t','pd','p','tz'],
        [ts,pds,ps,tzs]
    )
