import sys
sys.path.append('..')
import ConfigParser
import numpy as np
import os.path, time
import zmq
import sys
import pylab as pl
import config
import xcorrs
import tdoa
import sys
import heatMap
import argparse

SHOWHEAT = False

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
socket.connect ("tcp://localhost:%s" % config.getPortPublisher(0))
socket.setsockopt(zmq.SUBSCRIBE, "")

context2 = zmq.Context()
socket2 = context2.socket(zmq.SUB)
socket2.connect ("tcp://localhost:%s" % config.getPortPublisher(1))
socket2.setsockopt(zmq.SUBSCRIBE, "")

contextPub = zmq.Context()
socketPub = contextPub.socket(zmq.PUB)
socketPub.bind("tcp://*:%s" % config.getPortAnalysisPublisher())

# set up 
tdoa = tdoa.tdoa(grid_resolution = 800, doPhaseTransform = False, doBandpassFiltering = False)
if SHOWHEAT:
  heatMap = heatMap.heatMap(*tdoa.get_grid())

#calibrating sampling rate
print "calibrating sampling rates..."
rates = [config.getSamplingRate(bytearray(socket.recv())[:4]) for i in range(10)]
sampling_rate = np.median(rates)
print "min:%.4f max:%.4f median:%.4f, mean:%.4f, std:%.4f, touse:%.4f" %(np.min(rates), np.max(rates), np.median(rates), np.mean(rates), np.std(rates), sampling_rate)

tdoa.set_sampling_rate(sampling_rate)

def get_sig_from_socket(ss):
    string = bytearray(ss.recv())
    sigs = np.array(string[4:],dtype=np.float).reshape(LENG,4)
    assert sigs.shape == (LENG,4)
    return sigs

def energy_for_sig(s):
    a = s - np.mean(s)
    return np.mean(a * a)
    
while 1:
    sig1 = get_sig_from_socket(socket)
    sig2 = get_sig_from_socket(socket2)
    sys.stdout.flush()
    time_start = time.time()
    (maxx,maxy,r,theta,ll) = tdoa.calculate_liklihood_map([sig1,sig2])
    if SHOWHEAT:
      heatMap.update(ll, maxx,maxy)
    energy = max(energy_for_sig(sig1[:,0]),energy_for_sig(sig2[:,0]))
    print "dist: %.4f ang: %.4f time: %.4f energy:%.4f"% (r,theta,time.time() - time_start, energy)
    socketPub.send("%.6f %.6f %.6f" %(maxx,maxy, energy)); # publish result r, theta, energy

    sys.stdout.flush()
    clearQueue(socket)
    clearQueue(socket2)
