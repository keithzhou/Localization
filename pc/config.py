import struct 
class config():
    def __init__(self):
        array1 = {
          "MICS": ((-.22,0.0,0.0), (-.09,.18,0),(0.0, 0.0, 0.0),(0,0,0)), 
          "PORT_PASS_THROUGH": 5557, 
          "PORT_PUBLISHER": 5556, 
          "PORT_ANALYSIS": 5558,
          "SIDE": -1,
          "PORT_USB": '/dev/tty.usbmodem406861',
        }

        array2 = {
          "MICS": ((.24,0.0,0.0), (.0,.0,0),(0.095, 0.19, 0.0), (0,0,0)), 
          "PORT_PASS_THROUGH": 6557, 
          "PORT_PUBLISHER": 6556, 
          "PORT_ANALYSIS": 6558,
          "PORT_USB": '/dev/tty.usbmodem406691',
          "SIDE": 1
        }

        self.array_seperation = .3

        array1["MICS"] = [(i[0] + array1["SIDE"]*self.array_seperation/2.0, i[1], i[2]) for i in array1["MICS"]] 
        array2["MICS"] = [(i[0] + array2["SIDE"]*self.array_seperation/2.0,i[1],i[2]) for i in array2["MICS"]] 

        self.params_arrays = (array1, array2)
          
        self.DISTANCE_TEST = 0.4

        self.SPEED_SOUND = 343.8112863772779

        self.DATA_LENGTH = 6000

    def getPortUSB(self,array):
      return self.params_arrays[array]["PORT_USB"]

    def getPortPublisherPassThrough(self, array):
      return self.params_arrays[array]["PORT_PASS_THROUGH"]

    def getPortPublisher(self, array):
      return self.params_arrays[array]["PORT_PUBLISHER"]

    def getPortAnalysisPublisher(self, array):
      return self.params_arrays[array]["PORT_ANALYSIS"]

    def getMicLocs(self, array):
        return self.params_arrays[array]["MICS"]

    def getTestDistance(self):
        return self.DISTANCE_TEST

    def getSamplingRate(self, delay_bytes):
        return 1.0 /(struct.unpack("<L", delay_bytes)[0] * 1.0 / self.DATA_LENGTH *1e-6)

    def getSpeedSound(self):
        return self.SPEED_SOUND

    def getDataLength(self):
        return self.DATA_LENGTH

    def printDebug(self):
        print "mic locations:", self.getMicLocs(0)
        print "mic locations:", self.getMicLocs(1)
        print "test distance:", self.getTestDistance()
        print "speed of sound:", self.getSpeedSound()

if __name__ == "__main__":
    sut = config()
    sut.printDebug()
