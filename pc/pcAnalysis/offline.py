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


PATH = 'data'

CV = {
  'no_phase,no_bpf': {'grid_resolution': 800,'doPhaseTransform':False, 'doBandpassFiltering':False},
  'no_phase,no_bpf': {'grid_resolution': 400,'doPhaseTransform':False, 'doBandpassFiltering':False},
#  'no_phase,do_bpf': {'grid_resolution': 800,'doPhaseTransform':False, 'doBandpassFiltering':True},
#  'do_phase,no_bpf': {'grid_resolution': 800,'doPhaseTransform':True, 'doBandpassFiltering':False},
#  'do_phase,do_bpf': {'grid_resolution': 800,'doPhaseTransform':True, 'doBandpassFiltering':True},
}

for key in CV:
  print key, CV[key]
  engine = tdoa.tdoa(**CV[key])
  for f in os.listdir(PATH):
    m = re.search('rr_([\d\.]+)+_([\d\.-]+)\.p',f)
    r_gold = float(m.group(1)) / 100.0
    theta_gold = float(m.group(2))
    d = pickle.load(open(PATH + '/' + f,'rb'))
    thetas = []
    rs = []
    for data,freq in zip(d[0],d[1]):
      engine.set_sampling_rate(freq)
      (xx,yy,r,theta,ll) = engine.calculate_liklihood_map(data)
      rs.append(r)
      thetas.append(theta / 180.0 * np.pi)
      #print r,theta
    print "\tfile:",f,"r: %.2f theta: %.2f" %(r_gold,theta_gold),"error: %.4f" %np.mean(abs(np.array(rs) - r_gold))
