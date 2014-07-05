import serial
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import scipy.io.wavfile

USBPORTNAME = '/dev/tty.usbmodem406541'
USBBAUDRATE = 9600

def printUSB():
    ser = serial.Serial(USBPORTNAME, USBBAUDRATE)
    waveform = list()
    try:
        ts = datetime.datetime.now()
        while True:
            data = int(ser.readline())
            waveform.append(data)
            ttmp = datetime.datetime.now()
            print "data:%d since_last_message::%.6f" % (data, (ttmp - ts).microseconds/1e6)
            ts = ttmp
    except:
        waveform = np.array(waveform).astype(np.float) 
        waveform -= np.mean(waveform)
        waveform = (waveform / np.max(np.abs(waveform)) * 10000).astype(np.int16)
        fig = plt.figure()
        plt.plot(waveform)
        plt.savefig('output.png')
        print waveform.shape
        scipy.io.wavfile.write('output.wav',20000,waveform) # writing the sound to a file
        print "Done Saving"

if __name__ == "__main__":
    printUSB()
