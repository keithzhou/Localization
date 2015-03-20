import matplotlib.pyplot as plt
from pylab import cm

class heatMap():
  def __init__(self,xx,yy):
    (self.fig, self.ax, self.quad, self.dot) = self.createFig(xx,yy)

  def createFig(self,xx,yy):
    fig, ax = plt.subplots()
    quad = ax.pcolormesh(xx, yy, xx.T, cmap=cm.RdBu, vmin=0, vmax=3)
    cb = fig.colorbar(quad, ax=ax)
    dot, = ax.plot(.0,.0,'yo',markersize=10)
    plt.ion()
    plt.show()
    return fig,ax,quad,dot

  def update(self,data,dot_x,dot_y):
      plt.figure(self.fig.number)
      self.quad.set_clim(vmin=data.min(),vmax=abs(data).max())
      self.quad.set_array(data[:-1,:-1].ravel())
      self.dot.set_ydata(dot_y)
      self.dot.set_xdata(dot_x)
      plt.draw()

  def saveFig(self,name):
    plt.savefig(name)
