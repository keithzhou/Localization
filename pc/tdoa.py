import xcorrs
import config
import numpy as np
import operator

class tdoa():
  def __init__(self, sampling_rate = None, grid_resolution = 400, doBandpassFiltering = True, doPhaseTransform = True):
    self.sampling_rate = sampling_rate
    self.grid_resolution = grid_resolution
    self.doBandpassFiltering = doBandpassFiltering
    self.doPhaseTransform = doPhaseTransform
    self.config = config.config()
    self.xs = np.linspace(-.5,.5,self.grid_resolution)
    self.ys = np.linspace(-1.0,0.0,self.grid_resolution)
    (self.xx, self.yy) = np.meshgrid(self.xs,self.ys)


    self.ds = []
    for array in (0,1):
      (LOC_MIC1, LOC_MIC2, LOC_MIC3, LOC_MIC4) = self.config.getMicLocs(array)
      d1 = np.sqrt((self.xx - LOC_MIC1[0]) ** 2 + (self.yy - LOC_MIC1[1]) ** 2) 
      d2 = np.sqrt((self.xx - LOC_MIC2[0]) ** 2 + (self.yy - LOC_MIC2[1]) ** 2) 
      d3 = np.sqrt((self.xx - LOC_MIC3[0]) ** 2 + (self.yy - LOC_MIC3[1]) ** 2) 
      d4 = np.sqrt((self.xx - LOC_MIC4[0]) ** 2 + (self.yy - LOC_MIC4[1]) ** 2) 
      self.ds.append((d1,d2,d3,d4))

    if self.sampling_rate:
        self.sampling_rate_dependent_calculation()

  def get_grid(self):
    return (self.xx,self.yy)

  def set_sampling_rate(self, sampling_rate):
      self.sampling_rate = sampling_rate
      self.sampling_rate_dependent_calculation()

  def sampling_rate_dependent_calculation(self):
    SPEED_SOUND = self.config.getSpeedSound()
    self.ls = []
    for (d1,d2,d3,d4) in self.ds:
      # ti represents time(in terms of samples) from each point in the grid to the microphone
      t1 = d1 / SPEED_SOUND * self.sampling_rate
      t2 = d2 / SPEED_SOUND * self.sampling_rate
      t3 = d3 / SPEED_SOUND * self.sampling_rate
      t4 = d4 / SPEED_SOUND * self.sampling_rate
      l12 = np.rint(t1 - t2).astype(np.int)
      l13 = np.rint(t1 - t3).astype(np.int)
      l14 = np.rint(t1 - t4).astype(np.int)
      l23 = np.rint(t2 - t3).astype(np.int)
      l24 = np.rint(t2 - t4).astype(np.int)
      l34 = np.rint(t3 - t4).astype(np.int)
      self.ls.append((l12,l13,l14,l23,l24,l34))

  def ll_for_sigs(self,sigs, array):
    (sig1,sig2,sig3,sig4) = [sigs[:,i] for i in range(4)]
    xcorr12, xcorr13, xcorr14, xcorr23, xcorr24, xcorr34 = xcorrs.getXcorrs(sig1,sig2,sig3,sig4,self.sampling_rate, doBandpassFiltering = self.doBandpassFiltering, doPhaseTransform=self.doPhaseTransform)

#    xcorr12 = np.where(xcorr12 == np.max(xcorr12), 1, 0)
#    xcorr13 = np.where(xcorr13 == np.max(xcorr13), 1, 0)
#    xcorr23 = np.where(xcorr23 == np.max(xcorr23), 1, 0)

    maxloc12 = np.argmax(xcorr12) - (len(sig1) - 1)
    maxloc13 = np.argmax(xcorr13) - (len(sig1) - 1)
    maxloc14 = np.argmax(xcorr14) - (len(sig1) - 1)
    maxloc23 = np.argmax(xcorr23) - (len(sig1) - 1)
    maxloc24 = np.argmax(xcorr24) - (len(sig1) - 1)
    maxloc34 = np.argmax(xcorr34) - (len(sig1) - 1)
    l12 = self.ls[array][0]
    l13 = self.ls[array][1]
    l23 = self.ls[array][3]
    ll = (xcorr12[l12 + (len(sig1)-1)]) + (xcorr13[l13 + (len(sig1)-1)]) + (xcorr23[l23 + (len(sig1)-1)])
    #mm = np.max(ll)
    #ll = np.where(ll > 0.9 * mm, 1.0 , 0.0)
    return ll

  def calculate_liklihood_map(self,sigs):
    lls = [self.ll_for_sigs(s,i) for i,s in enumerate(sigs)]
    ll = reduce(operator.add,lls)
    (maxxx,maxyy) = np.where(ll == ll.max())
    maxx = self.xs[int(round(np.median(maxyy)))]
    maxy = self.ys[int(round(np.median(maxxx)))]
    r,theta = self.to_rth(maxx,maxy)
    return (maxx,maxy,r,theta,ll)

  def to_rth(self,maxx,maxy):
    ang = abs(np.arctan(maxy/maxx) / np.pi * 180)
    if (maxy > 0) and (maxx > 0):
        ang = ang * 1
    elif maxy > 0:
        ang = 180 - ang
    elif (maxy < 0) and (maxx > 0):
        ang = -1 * ang
    else:
        ang = -180 + ang
    return (np.sqrt(maxx**2+maxy**2), ang)
