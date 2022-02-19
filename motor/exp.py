import sys
sys.path.append('../utils')

import serial
import time
import ati
from NatNetClient import NatNetClient
import data
import numpy as np

CLIENT_ADDR = "192.168.1.188"
SERVER_ADDR = "192.168.1.166"

lm_list = []
updated = False
def receive_mocap_data(mocap_data):
    # print('Frame number: {}, Timestamp {}'.format(
    #     mocap_data.prefix_data.frame_number,
    #     mocap_data.suffix_data.timestamp
    # ))

    # for rb in mocap_data.rigid_body_data.rigid_body_list:
    #     print('Rigid body ID: {}, Position: {}, Rotation: {}'.format(
    #         rb.id_num, rb.pos, rb.rot
    #     ))

    # for lm in mocap_data.labeled_marker_data.labeled_marker_list:
    #     print('Marker ID: {} Position: {}'.format(
    #         lm.id_num,
    #         lm.pos
    #     ))

    global lm_list
    global updated

    lm_list = mocap_data.labeled_marker_data.labeled_marker_list
    updated = True

if __name__ == '__main__':
    streaming_client = NatNetClient()
    streaming_client.set_client_address(CLIENT_ADDR)
    streaming_client.set_server_address(SERVER_ADDR)
    streaming_client.set_use_multicast(True)
    streaming_client.set_print_level(0)
    streaming_client.mocap_data_listener = receive_mocap_data
    streaming_client.run()

    motor = serial.Serial()
    motor.port = '/dev/tty.usbserial-014343C7'
    motor.baudrate = 115200
    motor.open()

    time.sleep(2)

    t = []
    lm = {}

    jump = False
    tc = 0
    t0 = time.time()
    while tc < 2.5:

        if tc > 0.5 and not jump:
            motor.write((127).to_bytes(1,'big',signed=True))
            jump = True

        tc = time.time()-t0

        t.append(tc)
        if updated:
            for m in lm_list:
                if m.id_num not in lm:
                    lm[m.id_num] = {}
                    lm[m.id_num]['t'] = [tc]
                    lm[m.id_num]['pos'] = [m.pos]
                else:
                    lm[m.id_num]['t'].append(tc)
                    lm[m.id_num]['pos'].append(m.pos)
            updated = False

    motor.write((0).to_bytes(1,'big',signed=True))

    motor.close()
    streaming_client.shutdown()

    keys = []
    values = []

    for k in lm.keys():
        t = lm[k]['t']
        p = np.array(lm[k]['pos'])
        keys += ['t','x','y','z']
        values += [t,p[:,0],p[:,1],p[:,2]]

    file_name = '../data/test.csv'
    data.write(
        file_name,
        keys,
        values
    )
