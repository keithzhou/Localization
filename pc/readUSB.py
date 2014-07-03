import serial
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.io.wavfile

USBPORTNAME = '/dev/tty.usbmodem406541'
USBBAUDRATE = 9600

def printUSB():
    ser = serial.Serial(USBPORTNAME, USBBAUDRATE)
    waveform = list()
    try:
        while True:
            data = int(ser.readline())
            waveform.append(data)
            print "%d" % data
    except:
        print "Error"
        waveform = np.array(waveform) * 1.0
        waveform -= np.mean(waveform)
        waveform /= np.max(np.abs(waveform))
        waveform *= 32767
        scipy.io.wavfile.write('test.wav',44100,scaled)
        


if __name__ == "__main__":
    printUSB()
