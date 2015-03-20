import sys
sys.path.append('..')
import zmq
import config
import time
import struct

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("array")
args = parser.parse_args()
array = int(args.array)
print "using array:", array


config = config.config()
dataLength = config.getDataLength()

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:%s" % config.getPortPublisher(array))

contextrcv = zmq.Context()
socketrcv = context.socket(zmq.SUB)
socketrcv.connect ("tcp://localhost:%s" % config.getPortPublisherPassThrough(array))
socketrcv.setsockopt(zmq.SUBSCRIBE, "")

def printUSB():
    lastTime = time.time()
    while True:
        data = bytearray(socketrcv.recv())
        assert(len(data) == dataLength * 4 + 1 + 4)
        assert(data[-1] == ord('\n'))
        socket.send(data[:-1])
        print "time elapsed: %.4f sampling rate: %.4f" %(time.time() - lastTime, config.getSamplingRate(data[:4]))
        lastTime = time.time()

if __name__ == "__main__":
    printUSB()
