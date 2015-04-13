import sys
sys.path.append('..')
import zmq
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import config
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("array")
args = parser.parse_args()
array = int(args.array)
print "using array:", array

config = config.config()

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect ("tcp://localhost:%s" % config.getPortPublisher(array))

socket.setsockopt(zmq.SUBSCRIBE, "")

def plotChannels(waveform, fileName):
        f, axes = plt.subplots(waveform.shape[1], sharex=True, sharey=True)
        for i in range(waveform.shape[1]):
            axes[i].plot(waveform[:,i])
            axes[i].set_title("channel %d" % (i+1,))
        plt.savefig(fileName)

def processWaveform(waveform,SAMPLINGRATE):
    # save raw waveform
    #print "save raw array"
    plotChannels(waveform.T,'output_plot_raw.png')
    np.save("output_raw",waveform)

    # normalize waveform to save as audio file
    print "normalize for audio"
    waveform -= np.mean(waveform,axis=-1)[:,np.newaxis]
    waveform = (waveform / np.max(np.abs(waveform),axis=-1)[:,np.newaxis] * 10000).astype(np.int16)
    #plotChannels(waveform.T,'output_plot_audio.png')

    # writing the sound to a file
    print "save audio"
    scipy.io.wavfile.write('output_audio_ch1.wav',SAMPLINGRATE,waveform[0])
    scipy.io.wavfile.write('output_audio_ch2.wav',SAMPLINGRATE,waveform[1])
    scipy.io.wavfile.write('output_audio_ch3.wav',SAMPLINGRATE,waveform[2])
    scipy.io.wavfile.write('output_audio_ch4.wav',SAMPLINGRATE,waveform[3])

result = []
samplingRate = []
try:
    while 1:
        string = bytearray(socket.recv())
        current = np.array(string[4:],dtype=np.float).reshape(config.getDataLength(),4)
        samplingRate.append(config.getSamplingRate(string[:4]))
        print current.shape
        result.append(current)
except:
    pickle.dump([result,samplingRate], open( "save.p", "wb" ) )
    print "sampling rates:",np.median(samplingRate), "variation:", np.std(samplingRate)
    processWaveform(np.vstack(result).T,np.median(samplingRate))
