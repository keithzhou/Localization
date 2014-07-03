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
        ts = time.time()
        while True:
            data = int(ser.readline())
            waveform.append(data)
            print "%d dt:%.6f" % (data, time.time() - ts)
            ts = time.time()
    except:
        print "save audio file"
        waveform = np.array(waveform) * 1.0
        print "mean:%.4f" % np.mean(waveform)
        waveform -= np.mean(waveform)
        waveform = np.int16((waveform/np.max(np.abs(waveform))) * 32767)
        print waveform[:100]
        print "max:%d min:%d avg:%.4f" % (np.max(waveform), np.min(waveform), np.mean(waveform))
        scipy.io.wavfile.write('test.wav',100,waveform)

if __name__ == "__main__":
    printUSB()
