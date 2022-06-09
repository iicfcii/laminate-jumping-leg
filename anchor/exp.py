import matplotlib.pyplot as plt
import serial
import time
from utils import ati
from utils.NatNetClient import NatNetClient
from utils import data

CLIENT_ADDR = "192.168.1.188"
SERVER_ADDR = "192.168.1.166"

lm_pos = [0,0,0]
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

    global lm_pos
    global updated

    lm_list = mocap_data.labeled_marker_data.labeled_marker_list
    if len(lm_list) == 0: return
    lm_pos = lm_list[0].pos
    updated = True

jump_only = False

if __name__ == '__main__':
    if not jump_only:
        streaming_client = NatNetClient()
        streaming_client.set_client_address(CLIENT_ADDR)
        streaming_client.set_server_address(SERVER_ADDR)
        streaming_client.set_use_multicast(True)
        streaming_client.set_print_level(0)
        streaming_client.mocap_data_listener = receive_mocap_data
        streaming_client.run()
        sensor = ati.init()

    motor = serial.Serial()
    motor.port = '/dev/tty.usbserial-014343C7'
    motor.baudrate = 115200
    motor.open()

    time.sleep(2)

    t = []
    y = []
    grf = []

    jump = False
    tc = 0
    t0 = time.time()
    while tc < 1.5:
        if tc > 0.5 and not jump:
            motor.write((127).to_bytes(1,'big',signed=True))
            jump = True

        tc = time.time()-t0
        t.append(tc)

        if not jump_only:
            ati.single_read(sensor)
            f = ati.recv(sensor)
            grf.append(f[2])

            if updated:
                y.append(lm_pos[2])
                updated = False
            else:
                y.append(None)

    motor.write((0).to_bytes(1,'big',signed=True))

    motor.close()
    if not jump_only: streaming_client.shutdown()

    file_name = './data/test.csv'
    data.write(
        file_name,
        ['t','y','grf'],
        [t,y,grf]
    )
