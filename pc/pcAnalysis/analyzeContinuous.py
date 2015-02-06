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


LENG = config.getDataLength()# 10 to 12 ms
HISTLEN = 1

RESOLUTION_XY = 400
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

def xcorr_freq(s1,s2,SAMPLING_RATE):
    return xcorrs.xcorr_freq(s1,s2,SAMPLING_RATE)

def getXcorrs(sig1,sig2,sig3,sig4,SAMPLING_RATE):
    return xcorrs.getXcorrs(sig1,sig2,sig3,sig4,SAMPLING_RATE)

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

d1 = np.sqrt((xx - LOC_MIC1[0]) ** 2 + (yy - LOC_MIC1[1]) ** 2) #+ (zz - LOC_MIC1[2]) ** 2 ))
d2 = np.sqrt((xx - LOC_MIC2[0]) ** 2 + (yy - LOC_MIC2[1]) ** 2) #+ (zz - LOC_MIC1[2]) ** 2 ))
d3 = np.sqrt((xx - LOC_MIC3[0]) ** 2 + (yy - LOC_MIC3[1]) ** 2) #+ (zz - LOC_MIC1[2]) ** 2 ))
d4 = np.sqrt((xx - LOC_MIC4[0]) ** 2 + (yy - LOC_MIC4[1]) ** 2) #+ (zz - LOC_MIC1[2]) ** 2 ))

def buildMap2D(sig1,sig2,sig3,sig4,SAMPLING_RATE,figMap,ax,quad,dot):
    assert (len(sig1) == LENG)
    assert (len(sig2) == LENG)
    assert (len(sig3) == LENG)
    assert (len(sig4) == LENG)
    time_start = time.time()

#    xxx1, xxx2, xxx3, xxx4 = xcorrs.getXcorrsTemplate(sig1,sig2,sig3,sig4)

#    loc12 = np.argmax(xxx1) - np.argmax(xxx2)
#    loc13 = np.argmax(xxx1) - np.argmax(xxx3)
#    loc14 = np.argmax(xxx1) - np.argmax(xxx4)
#    loc23 = np.argmax(xxx2) - np.argmax(xxx3)
#    loc24 = np.argmax(xxx2) - np.argmax(xxx4)
#    loc34 = np.argmax(xxx3) - np.argmax(xxx4)
#
#    print loc12, loc13, loc14, loc23

    xcorr12, xcorr13, xcorr14, xcorr23, xcorr24, xcorr34 = getXcorrs(sig1,sig2,sig3,sig4,SAMPLING_RATE)
    maxloc12 = np.argmax(xcorr12) - (len(sig1) - 1)
    maxloc13 = np.argmax(xcorr13) - (len(sig1) - 1)
    maxloc14 = np.argmax(xcorr14) - (len(sig1) - 1)
    maxloc23 = np.argmax(xcorr23) - (len(sig1) - 1)
    maxloc24 = np.argmax(xcorr24) - (len(sig1) - 1)
    maxloc34 = np.argmax(xcorr34) - (len(sig1) - 1)
#    pa = sig1 - np.mean(sig1)
#    pb = sig2 - np.mean(sig2)
#    pc = sig3 - np.mean(sig3)
#    pd = sig4 - np.mean(sig4)
#    print "lag:12:%+.2f,13:%+.2f,14:%+.2f,23:%+.2f p1:%.2f,%.2f,%.2f,%.2f" % (maxloc12,maxloc13,maxloc14,maxloc23, np.sum(pa * pa)*1.0 / len(pa), np.sum(pb * pb) * 1.0 / len(pb),np.sum(pc * pc) * 1.0 / len(pc),np.sum(pd * pd) * 1.0 / len(pd))
#    fq12.addNum(maxloc12)
#    fq13.addNum(maxloc13)
#    fq14.addNum(maxloc14)
#    fq23.addNum(maxloc23)
#    fq24.addNum(maxloc24)
#    fq34.addNum(maxloc34)
    ll = (xcorr12[l12 + (len(sig1)-1)]) * (xcorr13[l13 + (len(sig1)-1)]) * (xcorr23[l23 + (len(sig1)-1)])
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
    #sock.send("msg:%.4f %.4f\n" % (np.sqrt(maxx**2+maxy**2),ang))

#    ll = xcorr12[l12] * xcorr13[l13] * xcorr23[l23] + xcorr14[l14]  + xcorr24[l24] + xcorr34[l34]
    updateFig(figMap,ax,quad,dot,ll, maxx,maxy)
    #plotXcorrDebug(xcorr12,xcorr13,xcorr14,xcorr23,xcorr24,xcorr34)
    print "dist: %.4f ang: %.4f time: %.4f"%(np.sqrt(maxx**2+maxy**2),ang,time.time() - time_start)

def xcorr(sig1, sig2, SAMPLING_RATE):
    a = (sig1 - np.mean(sig1))/np.std(sig1)
    b = (sig2 - np.mean(sig2))/np.std(sig2)
    #output = np.correlate(a,b,'full')
    output = xcorr_freq(a,b, SAMPLING_RATE)
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
                #print "Queue cleared"
                return
            else:
                raise

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect ("tcp://localhost:%s" % config.getPortPublisher())

socket.setsockopt(zmq.SUBSCRIBE, "")

ch1 = list()
ch2 = list()
ch3 = list()
ch4 = list()
figMap,ax, quad,dot = createFig()

#calibrating sampling rate
print "calibrating sampling rates..."
rates = [config.getSamplingRate(bytearray(socket.recv())[:4]) for i in range(10)]
sampling_rate = np.median(rates)
print "min:%.4f max:%.4f median:%.4f, mean:%.4f, std:%.4f, touse:%.4f" %(np.min(rates), np.max(rates), np.median(rates), np.mean(rates), np.std(rates), sampling_rate)

# sampling rate dependent calculations
t1 = d1 / SPEED_SOUND * sampling_rate
t2 = d2 / SPEED_SOUND * sampling_rate
t3 = d3 / SPEED_SOUND * sampling_rate
t4 = d4 / SPEED_SOUND * sampling_rate
l12 = np.rint(t1 - t2).astype(np.int)
l13 = np.rint(t1 - t3).astype(np.int)
l14 = np.rint(t1 - t4).astype(np.int)
l23 = np.rint(t2 - t3).astype(np.int)
l24 = np.rint(t2 - t4).astype(np.int)
l34 = np.rint(t3 - t4).astype(np.int)

while 1:
    string = bytearray(socket.recv())
    dd = np.array(string[4:],dtype=np.float).reshape(LENG,4)
    assert dd.shape == (LENG,4)
    sys.stdout.flush()
    buildMap2D(dd[:,0],dd[:,1],dd[:,2],dd[:,3],sampling_rate,figMap,ax,quad,dot)
    sys.stdout.flush()
    clearQueue(socket)
