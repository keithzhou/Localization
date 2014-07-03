import serial

ser = serial.Serial('/dev/tty.usbmodem406541',9600)
while True:
	print ser.readline()

