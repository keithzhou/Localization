import serial
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import scipy.io.wavfile

USBPORTNAME = '/dev/tty.usbmodem406941'
USBBAUDRATE = 9600

SAMPLINGRATE = 20000 * 5

def plotChannels(waveform, fileName):
        f, axes = plt.subplots(waveform.shape[1], sharex=True, sharey=True)
        for i in range(waveform.shape[1]):
            axes[i].plot(waveform[:,i])
            axes[i].set_title("channel %d" % (i+1,))
        plt.savefig(fileName)

def printUSB():
    ser = serial.Serial(USBPORTNAME, USBBAUDRATE)
    ch1 = list()
    ch2 = list()
    try:
        while True:
            data = ser.readline()
            df = data.split(' ')
            d1 = int(df[0])
            d2 = int(df[1])
            ch1.append(d1)
            ch2.append(d2)
            print "data:%d,%d" % (d1,d2) 
    except Exception, e:
        print repr(e)
    finally:
        waveform = np.array([ch1,ch2])
        # save raw waveform
        plotChannels(waveform.T,'output_plot_raw.png')

        # normalize waveform to save as audio file
        waveform = np.array(waveform).astype(np.float) 
        waveform -= np.mean(waveform,axis=-1)[:,np.newaxis]
        waveform = (waveform / np.max(np.abs(waveform),axis=-1)[:,np.newaxis] * 10000).astype(np.int16)
        plotChannels(waveform.T,'output_plot_audio.png')

        # writing the sound to a file
        scipy.io.wavfile.write('output_audio_ch1.wav',SAMPLINGRATE,waveform[0]) 
        scipy.io.wavfile.write('output_audio_ch2.wav',SAMPLINGRATE,waveform[1]) 
        print "Done Saving"

if __name__ == "__main__":
    printUSB()
