import sys
sys.path.append('..')
import zmq
import config

config = config.config()

PUBLISHERPORT = "5556"
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:%s" % config.getPortPublisher())

contextrcv = zmq.Context()
socketrcv = context.socket(zmq.SUB)
socketrcv.connect ("tcp://localhost:%s" % config.getPortPublisherPassThrough())
socketrcv.setsockopt(zmq.SUBSCRIBE, "")

def printUSB():
    last = list()
    while True:
        data = bytearray(socketrcv.recv())
        for i in data:
            if i == ord('\n'): # end of line detected
                if len(last) == 0:
                    pass
                elif len(last) == 4:
                    socket.send("DATA %s %s %s %s" % (last[0],last[1],last[2],last[3]))
                else:
                    print "len:",len(last), ''.join([chr(k) for k in last])
                last = list()
            else:
                last.append(i)

if __name__ == "__main__":
    printUSB()
