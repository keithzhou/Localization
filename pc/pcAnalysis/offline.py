import sys
sys.path.append('..')
import os
import pickle
import tdoa
import re
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time

PATH = 'saved'

R = []
for DL in range(600,6001,600):
  result = []
  for f in os.listdir(PATH):
    m1 = re.search('save_([-\d\.]+)+_([-\d\.-]+)_0\.p',f)
    if not m1:
      continue
    f2 = 'save_'+ m1.group(1)+ '_' + m1.group(2) + '_1.p'
#    print "use:", f, f2
    x_gold = float(m1.group(1)) / 100.0
    y_gold = float(m1.group(2)) / 100.0
    d = pickle.load(open(PATH + '/' + f,'rb'))
    d2 = pickle.load(open(PATH + '/' + f2,'rb'))
    d[1]["data_length"] = DL
    engine = tdoa.tdoa(d[1]['sampling_rate'], config = d[1], grid_resolution = 800, doPhaseTransform = True, doBandpassFiltering = True)

    engine.set_sampling_rate(d[1]['sampling_rate'])
    for s1,s2 in zip(d[0],d2[0]):
      t1 = time.time()
      (xx,yy,r,theta,ll) = engine.calculate_liklihood_map([s1[:DL],s2[:DL]])
      t2 = time.time()
      result.append([((xx-x_gold)**2 + (yy - y_gold)**2)**0.5, t2-t1])
#      print "%.4f,%.4f (%.4f,%.4f)" % (xx,yy,x_gold,y_gold)
#      print "\tfile:",f,"r: %.2f theta: %.2f" %(r_gold,theta_gold),"error: %.4f" %np.mean(abs(np.array(rs) - r_gold))
  toI = np.mean(result,axis=0)
  print "d_l: %d error: %.4f time: %.4f" % (DL, toI[0],toI[1])
  R.append(np.mean(result,axis=0))
