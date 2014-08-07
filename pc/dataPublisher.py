import serial
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import scipy.io.wavfile
import os
import zmq

USBPORTNAME = '/dev/tty.usbmodem406541'
USBBAUDRATE = 9600

PUBLISHERPORT = "5556"
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:%s" % PUBLISHERPORT)

def printUSB():
    ser = serial.Serial(USBPORTNAME, USBBAUDRATE)
    while True:
        data = ser.readline()
        fields = data.split();
        if (fields[0] == "D:"):
            socket.send("DATA %s %s %s %s" % (fields[1],fields[2], fields[3], fields[4]))
        else: 
            print data

if __name__ == "__main__":
    printUSB()
