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

PUBLISHERPORT = "5557"
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:%s" % PUBLISHERPORT)
def printUSB():
    ser = serial.Serial(USBPORTNAME, USBBAUDRATE)
    while True:
        data = ser.read(size=1000)
        socket.send(data)

if __name__ == "__main__":
    printUSB()
