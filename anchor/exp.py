import sys
sys.path.append('../utils')

import serial
import time
import ati
from NatNetClient import NatNetClient

def receive_mocap_data(mocap_data):
    print('Frame number: {}, Timestamp {}'.format(
        mocap_data.prefix_data.frame_number,
        mocap_data.suffix_data.timestamp
    ))

    for rb in mocap_data.rigid_body_data.rigid_body_list:
        print('Rigid body ID: {}, Position: {}, Rotation: {}'.format(
            rb.id_num, rb.pos, rb.rot
        ))

    for lm in mocap_data.labeled_marker_data.labeled_marker_list:
        print('Marker ID: {} Position: {}'.format(
            lm.id_num,
            lm.pos
        ))
    pass

CLIENT_ADDR = "192.168.1.188"
SERVER_ADDR = "192.168.1.166"

if __name__ == '__main__':
    streaming_client = NatNetClient()
    streaming_client.set_client_address(CLIENT_ADDR)
    streaming_client.set_server_address(SERVER_ADDR)
    streaming_client.set_use_multicast(True)
    streaming_client.set_print_level(0)
    streaming_client.mocap_data_listener = receive_mocap_data
    assert streaming_client.run(), 'Cannot start client.'
    time.sleep(1)
    assert streaming_client.connected(), 'Cannot connect.'
    print('connected')
# motor = serial.Serial()
# motor.port = '/dev/tty.usbserial-014343C7'
# motor.baudrate = 115200
# motor.open()
#
# sensor = ati.init()
#
# time.sleep(2)
#
# motor.write((0).to_bytes(1,'big',signed=True))
# t = 0
# t0 = time.time()
# while t < 5:
#     ati.single_read(sensor)
#     f = ati.recv(sensor)
#
#     t = time.time()-t0
#     print(t,f[2])
#     time.sleep(0.01)
#
#     # time.sleep(0.01)
# motor.write((0).to_bytes(1,'big',signed=True))
#
# motor.close()
