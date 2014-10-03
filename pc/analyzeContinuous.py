import ConfigParser
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft, fftfreq,fft,ifft
import os.path, time
import sys
from IPython.core.display import clear_output
import zmq
from mpl_toolkits.mplot3d import Axes3D
from pylab import cm
import sys
import pylab as pl
import RunningHist
import config

config = config.config()
(LOC_MIC1, LOC_MIC2, LOC_MIC3, LOC_MIC4) = config.getMicLocs()
SPEED_SOUND = config.getSpeedSound()
SAMPLING_RATE = config.getSamplingRate()


LENG = 2000 # 10 to 12 ms
HISTLEN = 1
xcorr_normalization = np.hstack([np.arange(LENG),np.arange(LENG-1)[::-1]]) + 1.0

RESOLUTION_XY = 1000
xs = np.linspace(-1.0,1.0,RESOLUTION_XY)
ys = np.linspace(-1.0,1.0,RESOLUTION_XY)
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
    assert(len(s1)==len(s2))
    pad = len(s1) * 2 - 1
    lenOld = len(s1)
    s1 = np.hstack([s1,np.zeros(pad)])
    s2 = np.hstack([s2,np.zeros(pad)])
    f_s1 = fft(s1)
    f_s2 = fft(s2)
    
    W = fftfreq(s1.size, 1.0 / SAMPLING_RATE)
    bolla = (abs(W) < .5e3) | (abs(W) > 7e3)
    f_s1[bolla] = 0.0
    f_s2[bolla] = 0.0
    
    f_s1c = np.conj(f_s1)
    f_s2c = np.conj(f_s2)
    f_s = f_s1 * f_s2c
    denom = np.abs(f_s)
    denom[denom < 1e-5] = 1e-5
    f_s = f_s / denom
    assert(np.any(np.isnan(f_s))==False)

    td = ifft(f_s)
    pos = td[:lenOld]
    neg = td[-lenOld+1:]
    return np.real(np.hstack([neg,pos]))

def getXcorrs(sig1,sig2,sig3,sig4):
    assert (len(sig1) == len(sig2))
    assert (len(sig2) == len(sig3))
    assert (len(sig3) == len(sig4))
    s1 = (sig1 - np.mean(sig1))/np.std(sig1)
    s2 = (sig2 - np.mean(sig2))/np.std(sig2)
    s3 = (sig3 - np.mean(sig3))/np.std(sig3)
    s4 = (sig4 - np.mean(sig4))/np.std(sig4)
    xcorr12 = xcorr_freq(s1,s2)
    xcorr13 = xcorr_freq(s1,s3)
    xcorr14 = xcorr_freq(s1,s4)
    xcorr23 = xcorr_freq(s2,s3)
    xcorr24 = xcorr_freq(s2,s4)
    xcorr34 = xcorr_freq(s3,s4)
    return (xcorr12, xcorr13, xcorr14, xcorr23, xcorr24, xcorr34)

def createFig():
    fig, ax = plt.subplots()
    quad = ax.pcolormesh(xx, yy, xx.T, cmap=cm.RdBu, vmin=0, vmax=3)
    cb = fig.colorbar(quad, ax=ax)
    dot, = ax.plot(.0,.0,'yo',markersize=10)
    plt.ion()
    plt.show()
    return fig,ax,quad,dot

def updateFig(figMap,ax,quad,dot,data):
    plt.figure(figMap.number)
    quad.set_clim(vmin=abs(data).min(),vmax=abs(data).max())
    quad.set_array(data[:-1,:-1].ravel())
    xa,ya = np.unravel_index(data.argmax(), data.shape)
    maxy = xs[xa]
    maxx = ys[ya]
    dot.set_ydata(maxy)
    dot.set_xdata(maxx)
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

l12 = (t1 - t2 + 0.5).astype(np.int)
#l12[l12 < 0] = 0
#l12[l12 > 2*LENG-1] = len(xcorr12) - 1

l13 = (t1 - t3 + 0.5).astype(np.int)
#l13[l13 < 0] = 0
#l13[l13 > len(xcorr12)-1] = len(xcorr12) - 1

l14 = (t1 - t4 + 0.5).astype(np.int)
#l14[l14 < 0] = 0
#l14[l14 > len(xcorr12)-1] = len(xcorr12) - 1

l23 = (t2 - t3 + 0.5).astype(np.int)
#l23[l23 < 0] = 0
#l23[l23 > len(xcorr12)-1] = len(xcorr12) - 1

l24 = (t2 - t4 + 0.5).astype(np.int)
#l24[l24 < 0] = 0
#l24[l24 > len(xcorr12)-1] = len(xcorr12) - 1

l34 = (t3 - t4 + 0.5).astype(np.int)
#l34[l34 < 0] = 0
#l34[l34 > len(xcorr12)-1] = len(xcorr12) - 1

def buildMap2D(sig1,sig2,sig3,sig4,figMap,ax,quad,dot):
    assert (len(sig1) == LENG)
    time_start = time.time()
    global offset_len 
    xcorr12, xcorr13, xcorr14, xcorr23, xcorr24, xcorr34 = getXcorrs(sig1,sig2,sig3,sig4)
    maxloc12 = np.argmax(xcorr12) - (len(sig1) - 1)
    maxloc13 = np.argmax(xcorr13) - (len(sig1) - 1)
    maxloc14 = np.argmax(xcorr14) - (len(sig1) - 1)
    maxloc23 = np.argmax(xcorr23) - (len(sig1) - 1)
    maxloc24 = np.argmax(xcorr24) - (len(sig1) - 1)
    maxloc34 = np.argmax(xcorr34) - (len(sig1) - 1)
    pa = sig1 - np.mean(sig1)
    pb = sig2 - np.mean(sig2)
    pc = sig3 - np.mean(sig3)
    pd = sig4 - np.mean(sig4)
    print "lag:12:%+.2f,13:%+.2f,14:%+.2f,23:%+.2f p1:%.2f,%.2f,%.2f,%.2f" % (maxloc12,maxloc13,maxloc14,maxloc23, np.sum(pa * pa)*1.0 / len(pa), np.sum(pb * pb) * 1.0 / len(pb),np.sum(pc * pc) * 1.0 / len(pc),np.sum(pd * pd) * 1.0 / len(pd))
    fq12.addNum(maxloc12)
    fq13.addNum(maxloc13)
    fq14.addNum(maxloc14)
    fq23.addNum(maxloc23)
    fq24.addNum(maxloc24)
    fq34.addNum(maxloc34)
    touse = [fq12.freqFor(l12),fq13.freqFor(l13),fq23.freqFor(l23)]
    touseTest = [abs(maxloc12),abs(maxloc13),abs(maxloc23)]
    touse.pop(np.argmax(touseTest))
#    ll = touse[0] + touse[1]
    ll = fq12.freqFor(l12) + fq13.freqFor(l13) + fq23.freqFor(l23)# + fq14.freqFor(l14) + fq24.freqFor(l24) + fq34.freqFor(l34)
    xa,ya = np.unravel_index(ll.argmax(), ll.shape)
    maxy = xs[xa]
    maxx = ys[ya]
    print "max loc:",maxx,maxy,np.sqrt(maxx**2+maxy**2),np.arctan(maxy/maxx)

#    ll = xcorr12[l12] * xcorr13[l13] * xcorr23[l23] + xcorr14[l14]  + xcorr24[l24] + xcorr34[l34]
    updateFig(figMap,ax,quad,dot,ll)
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

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect ("tcp://localhost:%s" % port)

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
