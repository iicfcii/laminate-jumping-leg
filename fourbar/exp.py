import time
import numpy as np
import matplotlib.pyplot as plt
from dynamixel_sdk import *

PORT_NAME = '/dev/tty.usbserial-FT5WJ6YV'
PROTOCOL_VERSION = 2.0
BAUDRATE = 1000000
SERVO_ID = 1

INIT_POS = 2048

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

# Init
rw(64,1,0) # Disable torque
rw(11,1,3) # Position mode
rw(64,1,1) # Enable torque
rw(116,4,INIT_POS) # Move to initial position
time.sleep(5)
p = rw(116,4)
print('Initial position', p)

# exit()

# PWM
rw(64,1,0) # Disable torque
rw(11,1,16) # PWM mode
rw(64,1,1) # Enable torque
rw(100,2,-885) # PWM

t0 = time.time()
t = 0
while t < 1:
    i = rw(126,2)
    p = rw(132,4)
    t = time.time()-t0
    print(t,i,p)

    # if p < INIT_POS-1000: rw(64,1,0) # Disable torque
    if t > 0.5: rw(64,1,0) # Disable torque

portH.closePort()
