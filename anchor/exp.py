import sys
sys.path.append('../utils')

import serial
import time
import ati
from NatNetClient import NatNetClient
import data


CLIENT_ADDR = "192.168.1.166"
SERVER_ADDR = "192.168.1.166"
lm_pos = [0,0,0]

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
    
    global lm_pos
    
    lm_list = mocap_data.labeled_marker_data.labeled_marker_list
    if len(lm_list) == 0: return
    lm_pos = lm_list[0].pos

if __name__ == '__main__':
    streaming_client = NatNetClient()
    streaming_client.set_client_address(CLIENT_ADDR)
    streaming_client.set_server_address(SERVER_ADDR)
    streaming_client.set_use_multicast(True)
    streaming_client.set_print_level(0)
    streaming_client.mocap_data_listener = receive_mocap_data
    streaming_client.run(), 'Cannot start client.'
    # streaming_client.connected(), 'Cannot connect.'

    motor = serial.Serial()
    motor.port = 'COM7'
    motor.baudrate = 115200
    motor.open()
    
    sensor = ati.init()
    
    time.sleep(2)

    t = []
    y = []
    grf = []

    motor.write((127).to_bytes(1,'big',signed=True))
    tc = 0
    t0 = time.time()
    while tc < 1:
        ati.single_read(sensor)
        f = ati.recv(sensor)
    
        tc = time.time()-t0
        
        t.append(tc)
        y.append(lm_pos[2])
        grf.append(f[2])
    
    motor.write((0).to_bytes(1,'big',signed=True))
    
    motor.close()
    streaming_client.shutdown()

    file_name = '../data/test.csv'
    data.write(
        file_name,
        ['t','y','grf'],
        [t,y,grf]
    )