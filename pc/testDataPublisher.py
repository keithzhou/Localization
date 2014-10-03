import serial
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import os
import zmq
import config

config = config.config()
(LOC_MIC1, LOC_MIC2, LOC_MIC3, LOC_MIC4) = config.getMicLocs()
SAMPLING_RATE = config.getSamplingRate()
DISTANCE_TEST = config.getTestDistance()
SPEED_SOUND = config.getSpeedSound()

def distanceBetween(l1,l2):
    a1 = np.array(l1)
    a2 = np.array(l2)
    r = a1 - a2
    return np.sqrt(r.dot(r))

def getDataAtTime(t):
    t = 5
    LOC_TEST = (DISTANCE_TEST * np.cos(2*np.pi*1.0/20*t), DISTANCE_TEST * np.sin(2*np.pi*1.0/20*t),0)
    print LOC_TEST[0], LOC_TEST[1], np.sqrt(LOC_TEST[0] ** 2 + LOC_TEST[1] ** 2), np.arctan(LOC_TEST[1]/LOC_TEST[0])

    d1 = distanceBetween(LOC_TEST,LOC_MIC1)
    d2 = distanceBetween(LOC_TEST,LOC_MIC2)
    d3 = distanceBetween(LOC_TEST,LOC_MIC3)
    d4 = distanceBetween(LOC_TEST,LOC_MIC4)

    t1 = d1 / SPEED_SOUND * SAMPLING_RATE
    t2 = d2 / SPEED_SOUND * SAMPLING_RATE
    t3 = d3 / SPEED_SOUND * SAMPLING_RATE
    t4 = d4 / SPEED_SOUND * SAMPLING_RATE

    t12 = int(t1 - t2 + 0.5)
    t13 = int(t1 - t3 + 0.5)
    t14 = int(t1 - t4 + 0.5)

    data = (np.random.rand(5000) * 256).astype(np.int)

    s1 = data
    s2 = np.roll(s1, -1 * t12)
    s3 = np.roll(s1, -1 * t13)
    s4 = np.roll(s1, -1 * t14)

    print t12,t13,t14
    return (s1,s2,s3,s4)

PUBLISHERPORT = "5556"
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:%s" % PUBLISHERPORT)

def send():
    count = 0
    while True:
        (s1,s2,s3,s4) = getDataAtTime(count)
        for i in range(len(s1)):
            socket.send("DATA %s %s %s %s" % (s1[i],s2[i],s3[i],s4[i]))
        count = count + 1
        time.sleep(5)
if __name__ == "__main__":
    send()
