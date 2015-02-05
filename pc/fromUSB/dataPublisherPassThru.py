import sys
sys.path.append('../')
import serial
import zmq
import config


config = config.config()

USBPORTNAME = '/dev/tty.usbmodem406541'
USBBAUDRATE = 9600

PUBLISHERPORT = config.getPortPublisherPassThrough()
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:%s" % PUBLISHERPORT)

def printUSB():
    ser = serial.Serial(USBPORTNAME, USBBAUDRATE)
    while True:
        data = ser.read(size=1000)
        socket.send(data)

if __name__ == "__main__":
    printUSB()
