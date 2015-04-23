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

PATH = 'saved'

CV = {
  '1': {'length': range(10,6001,100), 'doPhaseTransform': False, 'doBandpassFiltering':False},
}

result = []
for key in CV:
  print key, CV[key]
  for f in os.listdir(PATH):
    m1 = re.search('save_([-\d\.]+)+_([-\d\.-]+)_0\.p',f)
    if not m1:
      continue
    f2 = 'save_'+ m1.group(1)+ '_' + m1.group(2) + '_1.p'
    print "use:", f, f2
    x_gold = float(m1.group(1)) / 100.0
    y_gold = float(m1.group(2)) / 100.0
    d = pickle.load(open(PATH + '/' + f,'rb'))
    d2 = pickle.load(open(PATH + '/' + f2,'rb'))
    engine = tdoa.tdoa(d[1]['sampling_rate'], config = d[1], grid_resolution = 800, doPhaseTransform = False, doBandpassFiltering = False)

    for s1,s2 in zip(d[0],d2[0]):
      engine.set_sampling_rate(d[1]['sampling_rate'])
      (xx,yy,r,theta,ll) = engine.calculate_liklihood_map([s1,s2])
      result.append([xx,yy,x_gold,y_gold])
      print "%.4f,%.4f (%.4f,%.4f)" % (xx,yy,x_gold,y_gold)
#      print "\tfile:",f,"r: %.2f theta: %.2f" %(r_gold,theta_gold),"error: %.4f" %np.mean(abs(np.array(rs) - r_gold))

result = np.array(result)
result = np.mean(((result[:,2] - result[:,0]) ** 2 + (result[:,3] - result[:,1]) ** 2) ** 0.5)
print result
