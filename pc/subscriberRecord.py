import sys
import zmq
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile

SAMPLINGRATE = 50000

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect ("tcp://localhost:%s" % port)

topicfilter = "DATA"
socket.setsockopt(zmq.SUBSCRIBE, topicfilter)

def plotChannels(waveform, fileName):
        f, axes = plt.subplots(waveform.shape[1], sharex=True, sharey=True)
        for i in range(waveform.shape[1]):
            axes[i].plot(waveform[:,i])
            axes[i].set_title("channel %d" % (i+1,))
        plt.savefig(fileName)

def processWaveform(waveform):
    # save raw waveform
    plotChannels(waveform.T,'output_plot_raw.png')
    np.save("output_raw",waveform)

    # normalize waveform to save as audio file
    waveform -= np.mean(waveform,axis=-1)[:,np.newaxis]
    waveform = (waveform / np.max(np.abs(waveform),axis=-1)[:,np.newaxis] * 10000).astype(np.int16)
    plotChannels(waveform.T,'output_plot_audio.png')

    # writing the sound to a file
    scipy.io.wavfile.write('output_audio_ch1.wav',SAMPLINGRATE,waveform[0])
    scipy.io.wavfile.write('output_audio_ch2.wav',SAMPLINGRATE,waveform[1])
    scipy.io.wavfile.write('output_audio_ch3.wav',SAMPLINGRATE,waveform[2])
    scipy.io.wavfile.write('output_audio_ch4.wav',SAMPLINGRATE,waveform[3])
ch1 = list()
ch2 = list()
ch3 = list()
ch4 = list()
try:
    while 1:
        string = socket.recv()
        topic, d1, d2, d3, d4 = string.split()
        d1 = int(d1)
        d2 = int(d2)
        d3 = int(d3)
        d4 = int(d4)
        ch1.append(d1)
        ch2.append(d2)
        ch3.append(d3)
        ch4.append(d4)
except:
    waveform = np.array([ch1,ch2,ch3,ch4]).astype(np.float)
    processWaveform(waveform)
