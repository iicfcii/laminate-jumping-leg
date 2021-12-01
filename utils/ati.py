import numpy as np
import socket
import time

BYTE_ORDER = 'big'
ADDR = ('192.168.1.121',49152)

def bytes_to_str(bytes):
    return ' '.join(['{:02x}'.format(b) for b in bytes])

def init():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(2)
    return s

def start(s):
    msg = (0x1234).to_bytes(2,BYTE_ORDER)
    msg += (0x0002).to_bytes(2,BYTE_ORDER)
    msg += (0).to_bytes(4,BYTE_ORDER)
    s.sendto(msg,ADDR)

def stop(s):
    msg = (0x1234).to_bytes(2,BYTE_ORDER)
    msg += (0).to_bytes(2,BYTE_ORDER)
    msg += (0).to_bytes(4,BYTE_ORDER)
    s.sendto(msg,ADDR)

def single_read(s):
    msg = (0x1234).to_bytes(2,BYTE_ORDER)
    msg += (0x0002).to_bytes(2,BYTE_ORDER)
    msg += (1).to_bytes(4,BYTE_ORDER)
    s.sendto(msg,ADDR)

def set_bias(s):
    msg = (0x1234).to_bytes(2,BYTE_ORDER)
    msg += (0x0042).to_bytes(2,BYTE_ORDER)
    msg += (0).to_bytes(4,BYTE_ORDER)
    s.sendto(msg,ADDR)

def recv(s):
    data, addr = s.recvfrom(1024)
    f = []
    for i in range(6):
        f.append(int.from_bytes(data[12+4*i:12+4*i+4],byteorder=BYTE_ORDER,signed=True)/1000000)
    return f

if __name__=='__main__':
    sensor = init()
    set_bias(sensor)
    start(sensor)
    for i in range(1000):
        f = recv(sensor)
        print(f)
    stop(sensor)
