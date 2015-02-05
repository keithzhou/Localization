import sys
sys.path.append('..')
import ConfigParser
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft, fftfreq,fft,ifft
import os.path, time
from IPython.core.display import clear_output
import zmq
from mpl_toolkits.mplot3d import Axes3D
from pylab import cm
import sys
import pylab as pl
import RunningHist
import config
import xcorrs


#data
import socket
import sys

HOST, PORT = "localhost", 80

# Create a socket (SOCK_STREAM means a TCP socket)
#sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to server and send data
#sock.connect((HOST, PORT))
#sock.send("iam:array")
#end data

config = config.config()
(LOC_MIC1, LOC_MIC2, LOC_MIC3, LOC_MIC4) = config.getMicLocs()
SPEED_SOUND = config.getSpeedSound()
SAMPLING_RATE = config.getSamplingRate()


LENG = 2000 # 10 to 12 ms
HISTLEN = 1
xcorr_normalization = np.hstack([np.arange(LENG),np.arange(LENG-1)[::-1]]) + 1.0

RESOLUTION_XY = 300
xs = np.linspace(-.6,.6,RESOLUTION_XY)
ys = np.linspace(-.6,.6,RESOLUTION_XY)
zs = np.linspace(.0,.3,RESOLUTION_XY)
xx,yy = np.meshgrid(xs,ys)

fq12 = RunningHist.RunningHist(HISTLEN)
fq13 = RunningHist.RunningHist(HISTLEN)
fq14 = RunningHist.RunningHist(HISTLEN)
fq23 = RunningHist.RunningHist(HISTLEN)
fq24 = RunningHist.RunningHist(HISTLEN)
fq34 = RunningHist.RunningHist(HISTLEN)

def L2Between(loc1,loc2):
    ss = 0.0
    for i in range(3):
        ss += (loc1[i] - loc2[i]) ** 2
    return np.sqrt(ss)

def xcorr_freq(s1,s2):
    return xcorrs.xcorr_freq(s1,s2)

def getXcorrs(sig1,sig2,sig3,sig4):
    return xcorrs.getXcorrs(sig1,sig2,sig3,sig4)

def createFig():
    fig, ax = plt.subplots()
    quad = ax.pcolormesh(xx, yy, xx.T, cmap=cm.RdBu, vmin=0, vmax=3)
    cb = fig.colorbar(quad, ax=ax)
    dot, = ax.plot(.0,.0,'yo',markersize=10)
    plt.ion()
    plt.show()
    return fig,ax,quad,dot

def updateFig(figMap,ax,quad,dot,data,dot_x,dot_y):
    plt.figure(figMap.number)
    quad.set_clim(vmin=abs(data).min(),vmax=abs(data).max())
    quad.set_array(data[:-1,:-1].ravel())
    dot.set_ydata(dot_y)
    dot.set_xdata(dot_x)
    plt.draw()
 
plt.ion()
figaa = plt.figure()
ax12 = figaa.add_subplot(321)
ax13 = figaa.add_subplot(322)
ax13.set_title("1 vs 3")
ax14 = figaa.add_subplot(323)
ax23 = figaa.add_subplot(324)
ax24 = figaa.add_subplot(325)
ax34 = figaa.add_subplot(326)

def plotXcorrDebug(x12,x13,x14,x23,x24,x34):
    index_mid = int(len(x12) / 2)
    half_window = 50 
    plt.sca(ax12)
    plt.cla()
    ax12.plot(x12[index_mid - half_window: index_mid + half_window])

    plt.sca(ax13)
    plt.cla()
    ax13.plot(x13[index_mid - half_window: index_mid + half_window])

    plt.sca(ax14)
    plt.cla()
    ax14.plot(x14[index_mid - half_window: index_mid + half_window])

    plt.sca(ax23)
    plt.cla()
    ax23.plot(x23[index_mid - half_window: index_mid + half_window])

    plt.sca(ax24)
    plt.cla()
    ax24.plot(x24[index_mid - half_window: index_mid + half_window])

    plt.sca(ax34)
    plt.cla()
    ax34.plot(x34[index_mid - half_window: index_mid + half_window])
    plt.pause(0.0001) 

offset_len = LENG - 1
d1 = np.sqrt((xx - LOC_MIC1[0]) ** 2 + (yy - LOC_MIC1[1]) ** 2) #+ (zz - LOC_MIC1[2]) ** 2 ))
d2 = np.sqrt((xx - LOC_MIC2[0]) ** 2 + (yy - LOC_MIC2[1]) ** 2) #+ (zz - LOC_MIC1[2]) ** 2 ))
d3 = np.sqrt((xx - LOC_MIC3[0]) ** 2 + (yy - LOC_MIC3[1]) ** 2) #+ (zz - LOC_MIC1[2]) ** 2 ))
d4 = np.sqrt((xx - LOC_MIC4[0]) ** 2 + (yy - LOC_MIC4[1]) ** 2) #+ (zz - LOC_MIC1[2]) ** 2 ))

t1 = d1 / SPEED_SOUND * SAMPLING_RATE
t2 = d2 / SPEED_SOUND * SAMPLING_RATE
t3 = d3 / SPEED_SOUND * SAMPLING_RATE
t4 = d4 / SPEED_SOUND * SAMPLING_RATE

l12 = np.rint(t1 - t2).astype(np.int)
#l12[l12 < 0] = 0
#l12[l12 > 2*LENG-1] = len(xcorr12) - 1

l13 = np.rint(t1 - t3).astype(np.int)
#l13[l13 < 0] = 0
#l13[l13 > len(xcorr12)-1] = len(xcorr12) - 1

l14 = np.rint(t1 - t4).astype(np.int)
#l14[l14 < 0] = 0
#l14[l14 > len(xcorr12)-1] = len(xcorr12) - 1

l23 = np.rint(t2 - t3).astype(np.int)
#l23[l23 < 0] = 0
#l23[l23 > len(xcorr12)-1] = len(xcorr12) - 1

l24 = np.rint(t2 - t4).astype(np.int)
#l24[l24 < 0] = 0
#l24[l24 > len(xcorr12)-1] = len(xcorr12) - 1

l34 = np.rint(t3 - t4).astype(np.int)
#l34[l34 < 0] = 0
#l34[l34 > len(xcorr12)-1] = len(xcorr12) - 1

def buildMap2D(sig1,sig2,sig3,sig4,figMap,ax,quad,dot):
    assert (len(sig1) == LENG)
    time_start = time.time()
    global offset_len 
    xxx1, xxx2, xxx3, xxx4 = xcorrs.getXcorrsTemplate(sig1,sig2,sig3,sig4)

    loc12 = np.argmax(xxx1) - np.argmax(xxx2)
    loc13 = np.argmax(xxx1) - np.argmax(xxx3)
    loc14 = np.argmax(xxx1) - np.argmax(xxx4)
    loc23 = np.argmax(xxx2) - np.argmax(xxx3)
    loc24 = np.argmax(xxx2) - np.argmax(xxx4)
    loc34 = np.argmax(xxx3) - np.argmax(xxx4)

    print loc12, loc13, loc14, loc23

    xcorr12, xcorr13, xcorr14, xcorr23, xcorr24, xcorr34 = getXcorrs(sig1,sig2,sig3,sig4)
#    maxloc12 = np.argmax(xcorr12) - (len(sig1) - 1)
#    maxloc13 = np.argmax(xcorr13) - (len(sig1) - 1)
#    maxloc14 = np.argmax(xcorr14) - (len(sig1) - 1)
#    maxloc23 = np.argmax(xcorr23) - (len(sig1) - 1)
#    maxloc24 = np.argmax(xcorr24) - (len(sig1) - 1)
#    maxloc34 = np.argmax(xcorr34) - (len(sig1) - 1)
#    pa = sig1 - np.mean(sig1)
#    pb = sig2 - np.mean(sig2)
#    pc = sig3 - np.mean(sig3)
#    pd = sig4 - np.mean(sig4)
#    print "lag:12:%+.2f,13:%+.2f,14:%+.2f,23:%+.2f p1:%.2f,%.2f,%.2f,%.2f" % (maxloc12,maxloc13,maxloc14,maxloc23, np.sum(pa * pa)*1.0 / len(pa), np.sum(pb * pb) * 1.0 / len(pb),np.sum(pc * pc) * 1.0 / len(pc),np.sum(pd * pd) * 1.0 / len(pd))
    fq12.addNum(loc12)
    fq13.addNum(loc13)
    fq14.addNum(loc14)
    fq23.addNum(loc23)
    fq24.addNum(loc24)
    fq34.addNum(loc34)
    ll = xcorr12[l12 + (len(sig1)-1)] + xcorr13[l13 + (len(sig1)-1)] + xcorr23[l23 + (len(sig1)-1)]
    #ll = fq12.freqFor(l12) + fq13.freqFor(l13) + fq23.freqFor(l23)# + fq14.freqFor(l14) + fq24.freqFor(l24) + fq34.freqFor(l34)
#    xa,ya = np.unravel_index(ll.argmax(), ll.shape)
#    maxy = xs[xa]
#    maxx = ys[ya]
    (maxxx,maxyy) = np.where(ll == ll.max())
    maxx = xs[int(round(np.median(maxyy)))]
    maxy = ys[int(round(np.median(maxxx)))]
    ang = abs(np.arctan(maxy/maxx) / np.pi * 180)
    if (maxy > 0) and (maxx > 0):
        ang = ang * 1
    elif maxy > 0:
        ang = 180 - ang
    elif (maxy < 0) and (maxx > 0):
        ang = -1 * ang
    else:
        ang = -180 + ang
    print "max loc:",maxx,maxy,np.sqrt(maxx**2+maxy**2),ang
    #sock.send("msg:%.4f %.4f\n" % (np.sqrt(maxx**2+maxy**2),ang))

#    ll = xcorr12[l12] * xcorr13[l13] * xcorr23[l23] + xcorr14[l14]  + xcorr24[l24] + xcorr34[l34]
    updateFig(figMap,ax,quad,dot,ll, maxx,maxy)
    #plotXcorrDebug(xcorr12,xcorr13,xcorr14,xcorr23,xcorr24,xcorr34)
    print "time:", time.time() - time_start

def xcorr(sig1, sig2):
    a = (sig1 - np.mean(sig1))/np.std(sig1)
    b = (sig2 - np.mean(sig2))/np.std(sig2)
    #output = np.correlate(a,b,'full')
    output = xcorr_freq(a,b)
    return (np.argmax(output) - (len(sig2) - 1), output[np.argmax(output)]*1.0/len(sig1))

def bandpass_filtering(signal,fs,f_low,f_heigh):
    W = fftfreq(signal.size, 1.0 / fs)
    f_signal = rfft(signal)
    bolla = (abs(W) < f_low) | (abs(W) > f_heigh)
    f_signal[bolla] = 0.0
    return irfft(f_signal)

def clearQueue(s):
    while 1:
        try:
            s.recv(flags=zmq.NOBLOCK)
        except zmq.error.Again, e:
            if str("Resource temporarily unavailable") in repr(e):
                print "Queue cleared"
                return
            else:
                raise

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect ("tcp://localhost:%s" % config.getPortPublisher())

topicfilter = "DATA"
socket.setsockopt(zmq.SUBSCRIBE, topicfilter)

ch1 = list()
ch2 = list()
ch3 = list()
ch4 = list()
figMap,ax, quad,dot = createFig()
#figMap,ax, quad,dot = (None,None,None, None)
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
        #ch1_f = bandpass_filtering(ka,SAMPLING_RATE,1e3,5e3)
        #ch2_f = bandpass_filtering(kb,SAMPLING_RATE,1e3,5e3)
        #ch3_f = bandpass_filtering(kc,SAMPLING_RATE,1e3,5e3)
        #ch4_f = bandpass_filtering(kd,SAMPLING_RATE,1e3,5e3)
        #xcorrLag1, xcorrMag1 = xcorr(ch1, ch2)
        #xcorrLag2, xcorrMag2 = xcorr(ch1, ch3)
        #xcorrLag3, xcorrMag3 = xcorr(ch1, ch4)
        sys.stdout.flush()
        buildMap2D(ka,kb,kc,kd,figMap,ax,quad,dot)
        sys.stdout.flush()
        ch1 = list()
        ch2 = list()
        ch3 = list()
        ch4 = list()
        clearQueue(socket)
