import sys
sys.path.append('../utils')

import time
import numpy as np
import matplotlib.pyplot as plt
from dynamixel_sdk import *
import data

PORT_NAME = '/dev/tty.usbserial-FT5WJ6YV'
PROTOCOL_VERSION = 2.0
BAUDRATE = 1000000
SERVO_ID = 1

portH = PortHandler(PORT_NAME)
portH.setBaudRate(BAUDRATE)
packetH = PacketHandler(PROTOCOL_VERSION)

def rw(addr, l, val=None):
    if val is not None:
        groupSW = GroupSyncWrite(portH, packetH, addr, l)
        groupSW.addParam(SERVO_ID,(val).to_bytes(l,'little',signed=val < 0))
        groupSW.txPacket()

    groupSR = GroupSyncRead(portH, packetH, addr, l)
    groupSR.addParam(SERVO_ID)
    groupSR.txRxPacket()

    v = groupSR.getData(SERVO_ID,addr,l).to_bytes(l,'little')
    return int.from_bytes(v,'little',signed=True)

# PWM
rw(64,1,0) # Disable torque
rw(11,1,16) # PWM mode
rw(64,1,1) # Enable torque
dc = int(855/2)

T = []
I = []
V = []

t0 = time.time()

t = 0
i = 0
v = 0

rw(100,2,dc) # PWM
T.append(t)
I.append(i)
V.append(v)

while t < 1:
    t = time.time()-t0
    i = rw(126,2)
    v = rw(128,4)
    print(t,i,v)

    T.append(t)
    I.append(i)
    V.append(v)

rw(64,1,0)
portH.closePort()

file_name = '../data/motor_90_{}.csv'.format(dc)
data.write(
    file_name,
    ['t','i','v'],
    [T,I,V]
)
