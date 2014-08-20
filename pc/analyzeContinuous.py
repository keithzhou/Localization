import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, irfft, fftfreq
import os.path, time
import sys
from IPython.core.display import clear_output
import zmq
from mpl_toolkits.mplot3d import Axes3D
from pylab import cm
import sys
import pylab as pl

#constants

FILENAME = "output_raw.npy"

SPEED_SOUND = 340.0
SAMPLING_RATE = 50000
LENG = 500
xcorr_normalization = np.hstack([np.arange(LENG),np.arange(LENG-1)[::-1]]) + 1.0

LOC_MIC1 = ( 0.137,   0.0,     0.0)
LOC_MIC2 = (-0.06,  0.10392, 0.0)
LOC_MIC3 = (-0.0685, -0.11864, 0.0)
LOC_MIC4 = ( 0.0,    0.0,     0.145)

RESOLUTION_XY = 100 
xs = np.linspace(-.3,.3,RESOLUTION_XY)
ys = np.linspace(-.3,.3,RESOLUTION_XY)
yy,xx = np.meshgrid(xs,ys)

def L2Between(loc1,loc2):
    ss = 0.0
    for i in range(3):
        ss += (loc1[i] - loc2[i]) ** 2
    return np.sqrt(ss)

def getXcorrs(sig1,sig2,sig3,sig4):
    assert (len(sig1) == len(sig2))
    assert (len(sig2) == len(sig3))
    assert (len(sig3) == len(sig4))
    s1 = (sig1 - np.mean(sig1))/np.std(sig1)
    s2 = (sig2 - np.mean(sig2))/np.std(sig2)
    s3 = (sig3 - np.mean(sig3))/np.std(sig3)
    s4 = (sig4 - np.mean(sig4))/np.std(sig4)
    xcorr12 = np.correlate(s1,s2,'full') / xcorr_normalization
    xcorr13 = np.correlate(s1,s3,'full') / xcorr_normalization
    xcorr14 = np.correlate(s1,s4,'full') / xcorr_normalization
    xcorr23 = np.correlate(s2,s3,'full') / xcorr_normalization
    xcorr24 = np.correlate(s2,s4,'full') / xcorr_normalization
    xcorr34 = np.correlate(s3,s4,'full') / xcorr_normalization
    return (xcorr12, xcorr13, xcorr14, xcorr23, xcorr24, xcorr34)

def getXcorrs2D(sig1,sig2,sig3):
    assert (len(sig1) == len(sig2))
    assert (len(sig2) == len(sig3))
    s1 = (sig1 - np.mean(sig1))/np.std(sig1)
    s2 = (sig2 - np.mean(sig2))/np.std(sig2)
    s3 = (sig3 - np.mean(sig3))/np.std(sig3)
    xcorr12 = np.correlate(s1,s2,'full') / xcorr_normalization 
    xcorr13 = np.correlate(s1,s3,'full') / xcorr_normalization 
    xcorr23 = np.correlate(s2,s3,'full') / xcorr_normalization 
    return (xcorr12, xcorr13, xcorr23)

def createFig():
    fig, ax = plt.subplots()
    quad = ax.pcolormesh(xx, yy, xx, cmap=cm.RdBu, vmin=0, vmax=3)
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
    dot.set_xdata(xs[xa])
    dot.set_ydata(ys[ya])
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
    plt.sca(ax12)
    plt.cla()
    ax12.plot(x12[400:600])

    plt.sca(ax13)
    plt.cla()
    ax13.plot(x13[400:600])

    plt.sca(ax14)
    plt.cla()
    ax14.plot(x14[400:600])

    plt.sca(ax23)
    plt.cla()
    ax23.plot(x23[400:600])

    plt.sca(ax24)
    plt.cla()
    ax24.plot(x24[400:600])

    plt.sca(ax34)
    plt.cla()
    ax34.plot(x34[400:600])
    plt.pause(0.0001) 

def buildMap2D(sig1,sig2,sig3,sig4,figMap,ax,quad,dot):
    time_start = time.time()
    offset_len = len(sig1) - 1
    xcorr12, xcorr13, xcorr14, xcorr23, xcorr24, xcorr34 = getXcorrs(sig1,sig2,sig3,sig4)
    d1 = np.sqrt(((xx - LOC_MIC1[0]) ** 2 + (yy - LOC_MIC1[1]) ** 2))
    d2 = np.sqrt(((xx - LOC_MIC2[0]) ** 2 + (yy - LOC_MIC2[1]) ** 2))
    d3 = np.sqrt(((xx - LOC_MIC3[0]) ** 2 + (yy - LOC_MIC3[1]) ** 2))
    d4 = np.sqrt(((xx - LOC_MIC4[0]) ** 2 + (yy - LOC_MIC4[1]) ** 2))

    t1 = d1 / SPEED_SOUND * SAMPLING_RATE
    t2 = d2 / SPEED_SOUND * SAMPLING_RATE
    t3 = d3 / SPEED_SOUND * SAMPLING_RATE
    t4 = d4 / SPEED_SOUND * SAMPLING_RATE

    l12 = (t1 - t2 + offset_len + 0.5).astype(np.int)
    l12[l12 < 0] = 0
    l12[l12 > len(xcorr12)-1] = len(xcorr12) - 1

    l13 = (t1 - t3 + offset_len + 0.5).astype(np.int)
    l13[l13 < 0] = 0
    l13[l13 > len(xcorr12)-1] = len(xcorr12) - 1

    l14 = (t1 - t4 + offset_len + 0.5).astype(np.int)
    l14[l14 < 0] = 0
    l14[l14 > len(xcorr12)-1] = len(xcorr12) - 1

    l23 = (t2 - t3 + offset_len + 0.5).astype(np.int)
    l23[l23 < 0] = 0
    l23[l23 > len(xcorr12)-1] = len(xcorr12) - 1

    l24 = (t2 - t4 + offset_len + 0.5).astype(np.int)
    l24[l24 < 0] = 0
    l24[l24 > len(xcorr12)-1] = len(xcorr12) - 1

    l34 = (t3 - t4 + offset_len + 0.5).astype(np.int)
    l34[l34 < 0] = 0
    l34[l34 > len(xcorr12)-1] = len(xcorr12) - 1

    ll = xcorr12[l12] + xcorr13[l13] + xcorr14[l14] + xcorr23[l23] + xcorr24[l24] + xcorr34[l34]
    updateFig(figMap,ax,quad,dot,ll)
    #plotXcorrDebug(xcorr12,xcorr13,xcorr14,xcorr23,xcorr24,xcorr34)
    print "time:", time.time() - time_start


def buildMap(sig1, sig2, sig3, sig4):
    offset_len = len(sig1) - 1
    xcorr12, xcorr13, xcorr14, xcorr23, xcorr24, xcorr34 = getXcorrs(sig1,sig2,sig3,sig4)
    xs = np.linspace(-.2,.2,15)
    ys = np.linspace(-.2,.2,15)
    zs = np.linspace(0.0,.2,15)
    result_x = list()
    result_y = list()
    result_z = list()
    result_heat = list()
    for x in xs :
        for y in ys:
            for z in zs:
                ll = 0.0
                loc_current = (x,y,z)
                d1 = L2Between(loc_current,LOC_MIC1)
                d2 = L2Between(loc_current,LOC_MIC2)
                d3 = L2Between(loc_current,LOC_MIC3)
                d4 = L2Between(loc_current,LOC_MIC4)
                t1 = d1 / SPEED_SOUND * SAMPLING_RATE
                t2 = d2 / SPEED_SOUND * SAMPLING_RATE
                t3 = d3 / SPEED_SOUND * SAMPLING_RATE
                t4 = d4 / SPEED_SOUND * SAMPLING_RATE
                l12 = max(min(int(t1 - t2 + offset_len + 0.5),len(xcorr12)-1),0)
                l13 = max(min(int(t1 - t3 + offset_len + 0.5),len(xcorr12)-1),0)
                l14 = max(min(int(t1 - t4 + offset_len + 0.5),len(xcorr12)-1),0)
                l23 = max(min(int(t2 - t3 + offset_len + 0.5),len(xcorr12)-1),0)
                l24 = max(min(int(t2 - t4 + offset_len + 0.5),len(xcorr12)-1),0)
                l34 = max(min(int(t3 - t4 + offset_len + 0.5),len(xcorr12)-1),0)
                ll += xcorr12[l12] + xcorr13[l13] + xcorr14[l14] + xcorr23[l23] + xcorr24[l24] + xcorr34[l34]
                result_x.append(x)
                result_y.append(y)
                result_z.append(z)
#                result_heat.append(np.sqrt(x**2 + y**2 + z**2))
                result_heat.append(ll)
#                result_heat.append(d4)
#                result_heat.append(1.0 * l14 * 1.0)
    plotHeat(result_x,result_y,result_z,np.array(result_heat))

def plotHeat(xs,ys,zs,the_fourth_dimension):
#    colors = cm.hsv(the_fourth_dimension/max(the_fourth_dimension))
    colors = cm.hsv((the_fourth_dimension - min(the_fourth_dimension))/(max(the_fourth_dimension) - min(the_fourth_dimension)) * .6 + .2)
    colmap = cm.ScalarMappable(cmap=cm.hsv)
    colmap.set_array(the_fourth_dimension)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111,projection='3d')

    yg = ax.scatter(xs, ys, zs, c=colors, marker='o')
    cb = fig.colorbar(colmap)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

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
count = 0;
figMap,ax, quad,dot = createFig()
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
        ch1_f = bandpass_filtering(ka,SAMPLING_RATE,1e3,5e3)
        ch2_f = bandpass_filtering(kb,SAMPLING_RATE,1e3,5e3)
        ch3_f = bandpass_filtering(kc,SAMPLING_RATE,1e3,5e3)
        ch4_f = bandpass_filtering(kd,SAMPLING_RATE,1e3,5e3)
        xcorrLag1, xcorrMag1 = xcorr(ch1_f, ch2_f)
        xcorrLag2, xcorrMag2 = xcorr(ch1_f, ch3_f)
        xcorrLag3, xcorrMag3 = xcorr(ch1_f, ch4_f)
        count += 1
        pa = ka - np.mean(ka)
        pb = kb - np.mean(kb)
        pc = kc - np.mean(kc)
        pd = kd - np.mean(kd)
        print "lag:%+.2f,%+.2f,%+.2f p1:%.2f,%.2f,%.2f,%.2f" % (xcorrLag1,xcorrLag2,xcorrLag3, np.sum(pa * pa)*1.0 / len(pa), np.sum(pb * pb) * 1.0 / len(pb),np.sum(pc * pc) * 1.0 / len(pc),np.sum(pd * pd) * 1.0 / len(pd))
        sys.stdout.flush()
        buildMap2D(ch1_f,ch2_f,ch3_f,ch4_f,figMap,ax,quad,dot)
#        buildMap(ch1_f,ch2_f,ch3_f,ch4_f)
        ch1 = list()
        ch2 = list()
        ch3 = list()
        ch4 = list()
        clearQueue(socket)
        
