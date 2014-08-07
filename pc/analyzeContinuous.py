import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft, fftfreq
import os.path, time
import sys
from IPython.core.display import clear_output
import zmq

#constants

FILENAME = "output_raw.npy"

SPEED_SOUND = 340
SAMPLING_RATE = 20000
DISTANCE_SPEAKER = .12


def xcorr(sig1, sig2):
    a = (sig1 - np.mean(sig1))/np.std(sig1)
    b = (sig2 - np.mean(sig2))/np.std(sig2)
    output = np.correlate(a,b,'full')
    return (np.argmax(output) - (len(sig2) - 1), output[np.argmax(output)]*1.0/len(sig1))

def bandpass_filtering(signal,fs,f_low,f_heigh):
    W = fftfreq(signal.size, 1.0 / fs)
    f_signal = rfft(signal)
    bolla = (abs(W) < f_low) | (abs(W) > f_heigh)
    f_signal[bolla] = 0.0
    return irfft(f_signal)

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect ("tcp://localhost:%s" % port)

topicfilter = "DATA"
socket.setsockopt(zmq.SUBSCRIBE, topicfilter)

LENG = 1000
ch1 = list()
ch2 = list()
ch3 = list()
ch4 = list()
count = 0;
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
    if len(ch1) >= LENG:
        ka = np.array(ch1).astype(np.float)
        kb = np.array(ch2).astype(np.float)
        kc = np.array(ch3).astype(np.float)
        kd = np.array(ch4).astype(np.float)
        ch1_f = bandpass_filtering(ka,SAMPLING_RATE,0.2e3,50e3)
        ch2_f = bandpass_filtering(kb,SAMPLING_RATE,0.2e3,50e3)
        ch3_f = bandpass_filtering(kc,SAMPLING_RATE,0.2e3,50e3)
        ch4_f = bandpass_filtering(kd,SAMPLING_RATE,0.2e3,50e3)
        xcorrLag1, xcorrMag1 = xcorr(ch1_f, ch2_f)
        xcorrLag2, xcorrMag2 = xcorr(ch1_f, ch3_f)
        xcorrLag3, xcorrMag3 = xcorr(ch1_f, ch4_f)
        count += 1
        pa = ka - np.mean(ka)
        pb = kb - np.mean(kb)
        pc = kc - np.mean(kc)
        pd = kd - np.mean(kd)
        print "lag:%+.2f,%+.2f,%+.2f p1:%.2f,%.2f,%.2f,%.2f" % (xcorrLag1,xcorrLag2,xcorrLag3, np.sum(pa * pa)*1.0 / len(pa), np.sum(pb * pb) * 1.0 / len(pb),np.sum(pc * pc) * 1.0 / len(pc),np.sum(pd * pd) * 1.0 / len(pd))
        ch1 = list()
        ch2 = list()
        ch3 = list()
        ch4 = list()
