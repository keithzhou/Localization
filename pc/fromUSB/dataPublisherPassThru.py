import sys
sys.path.append('../')
import serial
import zmq
import config


config = config.config()
DATA_LENGTH = config.getDataLength()

USBPORTNAME = '/dev/tty.usbmodem406541'
USBBAUDRATE = 9600

PUBLISHERPORT = config.getPortPublisherPassThrough()
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:%s" % PUBLISHERPORT)

def printUSB():
    ser = serial.Serial(USBPORTNAME, USBBAUDRATE)
    data = ser.read(size=DATA_LENGTH*4*10) # empty buffer
    data = ser.read(size=6000*4 + 1 + 4) # 4 bytes for time elapse and 1 byte for '\n'
    fields = data.split('\n')
    assert len(fields) == 2
    print "field length:", [len(i) for i in fields]
    ser.read(size=len(fields[0]) + 1)
    while True:
        data = ser.read(size=DATA_LENGTH*4 + 1 + 4)
        socket.send(data)

if __name__ == "__main__":
    printUSB()
