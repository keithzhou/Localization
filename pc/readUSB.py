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
        waveform = np.array(waveform).astype(np.float) 
        waveform = (((waveform - np.mean(waveform)) / np.max(np.abs(waveform))) * 10000).astype(np.int16)
        scipy.io.wavefile.write('output.wav',44100,waveform) # writing the sound to a file
        print "Done Saving"

if __name__ == "__main__":
    printUSB()
