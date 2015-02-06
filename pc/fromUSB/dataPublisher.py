import sys
sys.path.append('..')
import zmq
import config
import time
import struct

config = config.config()
dataLength = config.getDataLength()

PUBLISHERPORT = "5556"
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:%s" % config.getPortPublisher())

contextrcv = zmq.Context()
socketrcv = context.socket(zmq.SUB)
socketrcv.connect ("tcp://localhost:%s" % config.getPortPublisherPassThrough())
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
