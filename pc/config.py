import struct 
class config():
    def __init__(self):
        array1 = {
          "MICS": ((-.22,0.0,0.0), (-.09,.18,0),(0.0, 0.0, 0.0),(0,0,0)), 
          "PORT_PASS_THROUGH": 5557, 
          "PORT_PUBLISHER": 5556, 
          "SIDE": -1,
          "PORT_USB": '/dev/tty.usbmodem406861',
        }

        array2 = {
          "MICS": ((.24,0.0,0.0), (.0,.0,0),(0.095, 0.19, 0.0), (0,0,0)), 
          "PORT_PASS_THROUGH": 6557, 
          "PORT_PUBLISHER": 6556, 
          "PORT_USB": '/dev/tty.usbmodem406691',
          "SIDE": 1
        }

        self.array_seperation = 1.0

        array1["MICS"] = [(i[0] + array1["SIDE"]*self.array_seperation/2.0, i[1], i[2]) for i in array1["MICS"]] 
        array2["MICS"] = [(i[0] + array2["SIDE"]*self.array_seperation/2.0,i[1],i[2]) for i in array2["MICS"]] 

        self.params_arrays = (array1, array2)
          
        self.DISTANCE_TEST = 0.4

        self.SPEED_SOUND = 343.8112863772779

        self.param_port_analysis = 6558  #analysis result will be published through this port

        self.DATA_LENGTH = 6000

    def getConfig(self): # return a dictionary of config data
      result = {
        "arrays": self.params_arrays,
        "speed_sound": self.SPEED_SOUND,
        "data_length": self.DATA_LENGTH,
        "sampling_rate": None
      }
      return result

    def getPortUSB(self,array):
      return self.params_arrays[array]["PORT_USB"]

    def getPortPublisherPassThrough(self, array):
      return self.params_arrays[array]["PORT_PASS_THROUGH"]

    def getPortPublisher(self, array):
      return self.params_arrays[array]["PORT_PUBLISHER"]

    def getPortAnalysisPublisher(self):
      return self.param_port_analysis

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
      c = self.getConfig()
      print c

if __name__ == "__main__":
    sut = config()
    sut.printDebug()
