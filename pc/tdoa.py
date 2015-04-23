import xcorrs
import time
import config
import numpy as np
import operator
from scipy.fftpack import rfft, irfft, fftfreq,fft,ifft

# will improve this file

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
    self.dataLength = self.config.getDataLength()

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
      l12 = np.rint(t1 - t2).astype(np.int) + self.dataLength - 1
      l13 = np.rint(t1 - t3).astype(np.int) + self.dataLength - 1
      l14 = np.rint(t1 - t4).astype(np.int) + self.dataLength - 1
      l23 = np.rint(t2 - t3).astype(np.int) + self.dataLength - 1
      l24 = np.rint(t2 - t4).astype(np.int) + self.dataLength - 1
      l34 = np.rint(t3 - t4).astype(np.int) + self.dataLength - 1

      print "l12(%d,%d), l13(%d,%d), l23(%d,%d)"%(np.min(l12),np.max(l12),np.min(l13),np.max(l13),np.min(l23),np.max(l23))

      self.ls.append((l12,l13,l14,l23,l24,l34))

  def arg_max_corr(self,a, b):

      if len(a.shape) > 1:
          raise ValueError('Needs a 1-dimensional array.')

      length = len(a)
      if not length % 2 == 0:
          raise ValueError('Needs an even length array.')

      if not a.shape == b.shape:
          raise ValueError('The 2 arrays need to be the same shape')


      omega = np.zeros(length)
      omega[0:length/2] = (2*np.pi*np.arange(length/2))/length
      omega[length/2+1:] = (2*np.pi*
              (np.arange(length/2+1, length)-length))/length

      fft_a = fft(a)
      def correlate_point(tau):
          rotate_vec = np.exp(1j*tau*omega)
          rotate_vec[length/2] = np.cos(np.pi*tau)
          return np.sum((ifft(fft_a*rotate_vec)).real*b)

      return np.vectorize(correlate_point)

  def ll_for_sigs(self,sigs, array):
    (sig1,sig2,sig3,sig4) = [sigs[:,i] for i in range(4)]
    t0 = time.time()
    xcorr12, xcorr13, xcorr14, xcorr23, xcorr24, xcorr34 = xcorrs.getXcorrs(sig1,sig2,sig3,sig4,self.sampling_rate, doBandpassFiltering = self.doBandpassFiltering, doPhaseTransform=self.doPhaseTransform)
    t1 = time.time()

#    ff12 = self.arg_max_corr(sig1,sig2)
#    ff13 = self.arg_max_corr(sig1,sig3)
#    ff23 = self.arg_max_corr(sig2,sig3)

#    xcorr12 = np.where(xcorr12 == np.max(xcorr12), 1, 0)
#    xcorr13 = np.where(xcorr13 == np.max(xcorr13), 1, 0)
#    xcorr23 = np.where(xcorr23 == np.max(xcorr23), 1, 0)

    l12 = self.ls[array][0]
    l13 = self.ls[array][1]
    l23 = self.ls[array][3]
    t2 = time.time()

#    print "a12"
#    aa12 = ff12(l12)
#    print "a13"
#    aa13 = ff13(l13)
#    print "a23"
#    aa23 = ff23(l23)
#
#    ll = aa12 + aa13 + aa23
#
    ll = (xcorr12[l12] ) + (xcorr13[l13]) +(xcorr23[l23]) 
    t3 = time.time()
    #print "get_map: xcorr:%.4f access:%.4f lookup:%.4f" % (t1 - t0, t2-t1, t3 - t2)
    #mm = np.max(ll)
    #ll = np.where(ll > 0.9 * mm, 1.0 , 0.0)
    return ll

  def calculate_liklihood_map(self,sigs):
    t0 = time.time()
    lls = [self.ll_for_sigs(s,i) for i,s in enumerate(sigs)]
    t1 = time.time()
    ll = reduce(operator.mul,lls)
    t2 = time.time()
    (maxxx,maxyy) = np.where(ll == ll.max())
    maxx = self.xs[int(round(np.median(maxyy)))]
    maxy = self.ys[int(round(np.median(maxxx)))]
    r,theta = self.to_rth(maxx,maxy)
    t3 = time.time()
    #print "get_map:%.4f reduce:%.4f other:%.4f" % (t1 - t0, t2 - t1, t3- t2)
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
