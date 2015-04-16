import sys
sys.path.append('..')
import config
import zmq 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D


config = config.config()

context2 = zmq.Context()
socket2 = context2.socket(zmq.SUB)
socket2.connect ("tcp://localhost:%s" % config.getPortAnalysisPublisher())
socket2.setsockopt(zmq.SUBSCRIBE, "")

class Scope:
    def __init__(self, ax,):
        self.ax = ax
        self.xdata = []
        self.ydata = []
        self.dot, = ax.plot(self.xdata,self.ydata,'go',markersize=5)

        self.ax.set_ylim(-1.0, 0)
        self.ax.set_xlim(-0.5, 0.5)
        self.count = 0

    def update(self):
        data = socket2.recv()
        (x,y,energy) = [float(i) for i in data.split()]
        if energy > 20.0:
          self.xdata += [x]
          self.ydata += [y]
          self.dot.set_ydata(self.ydata)
          self.dot.set_xdata(self.xdata)
          #plt.savefig("result_%05d.png" % self.count)
          self.count += 1
          plt.draw()

fig, ax = plt.subplots()
scope = Scope(ax)
plt.ion()
plt.show()

while 1:
  scope.update()
