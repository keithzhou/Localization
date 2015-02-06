import sys
sys.path.append('..')
import ConfigParser
import numpy as np
import os.path, time
import zmq
import sys
import pylab as pl
import RunningHist
import config
import xcorrs
import tdoa
import socket
import sys
import heatMap

tdoa = tdoa.tdoa(grid_resolution = 400, doPhaseTransform = False, doBandpassFiltering = False)

HOST, PORT = "localhost", 80

# Create a socket (SOCK_STREAM means a TCP socket)
#sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to server and send data
#sock.connect((HOST, PORT))
#sock.send("iam:array")
#end data

HISTLEN = 1

fq12 = RunningHist.RunningHist(HISTLEN)
fq13 = RunningHist.RunningHist(HISTLEN)
fq14 = RunningHist.RunningHist(HISTLEN)
fq23 = RunningHist.RunningHist(HISTLEN)
fq24 = RunningHist.RunningHist(HISTLEN)
fq34 = RunningHist.RunningHist(HISTLEN)

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

config = config.config()
LENG = config.getDataLength()# 10 to 12 ms
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect ("tcp://localhost:%s" % config.getPortPublisher())
socket.setsockopt(zmq.SUBSCRIBE, "")

#figMap,ax, quad,dot = createFig(*tdoa.get_grid())
heatMap = heatMap.heatMap(*tdoa.get_grid())

#calibrating sampling rate
print "calibrating sampling rates..."
rates = [config.getSamplingRate(bytearray(socket.recv())[:4]) for i in range(10)]
sampling_rate = np.median(rates)
print "min:%.4f max:%.4f median:%.4f, mean:%.4f, std:%.4f, touse:%.4f" %(np.min(rates), np.max(rates), np.median(rates), np.mean(rates), np.std(rates), sampling_rate)

tdoa.set_sampling_rate(sampling_rate)

while 1:
    string = bytearray(socket.recv())
    sigs = np.array(string[4:],dtype=np.float).reshape(LENG,4)
    assert sigs.shape == (LENG,4)
    sys.stdout.flush()
    time_start = time.time()
    (maxx,maxy,r,theta,ll) = tdoa.calculate_liklihood_map(sigs)
    #sock.send("msg:%.4f %.4f\n" % (np.sqrt(maxx**2+maxy**2),ang))
    heatMap.update(ll, maxx,maxy)
    print "dist: %.4f ang: %.4f time: %.4f"%(r,theta,time.time() - time_start)
    sys.stdout.flush()
    clearQueue(socket)
