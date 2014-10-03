import serial
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import scipy.io.wavfile
import os
import zmq
import struct

USBPORTNAME = '/dev/tty.usbmodem406541'
USBBAUDRATE = 9600

PUBLISHERPORT = "5556"
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:%s" % PUBLISHERPORT)

port = "5557"
contextrcv = zmq.Context()
socketrcv = context.socket(zmq.SUB)
socketrcv.connect ("tcp://localhost:%s" % port)
socketrcv.setsockopt(zmq.SUBSCRIBE, "")

def printUSB():
    last = list()
    while True:
        data = bytearray(socketrcv.recv())
#        data = bytearray(ser.read(size=100))
        for i in data:
            if i == ord('\n'): # end of line detected
                if len(last) == 0:
                    pass
                elif len(last) == 4:
                    socket.send("DATA %s %s %s %s" % (last[0],last[1],last[2],last[3]))
                else:
                    print "len:",len(last), ''.join([chr(k) for k in last])
                last = list()
            else:
                last.append(i)
if __name__ == "__main__":
    printUSB()
