import pandas as pd
import numpy as np
from scipy.stats import norm

def GetAccuracies(metapd : pd.DataFrame, get_locs : dict = None, ign_locs : dict = None ):
  '''Gets accuracies and counts number of correct/wrong answers.
  Includes only get_locs column values if given, and exludes ign_locs.
  e.g., if get_locs={'target_color':'yellow'}, only points with target
  color 'yellow' will be considered.

  Returns 
  -------
  accuracy : float

  : list of int
    list of true positives, true negatives, false positives, false negatives
  '''
  metacopy = metapd.copy()
  if not (get_locs is None):
    for key,val in get_locs.items():
      metacopy = metacopy[metacopy[key]==val]
  if not (ign_locs is None):
    for key,val in ign_locs.items():
      metacopy = metacopy[metacopy[key]!=val]
  
  accuracy = metacopy['isright'].sum()/len(metacopy)
  truepos = metacopy[metacopy['has_target']==True]['isright'].sum()
  trueneg = metacopy[metacopy['has_target']==False]['isright'].sum()
  falsepos = (1-metacopy[(metacopy['has_target']==False)]['isright']).sum()
  falseneg = (1-metacopy[(metacopy['has_target']==True)]['isright']).sum()
  return accuracy,[truepos,trueneg,falsepos,falseneg]

def GetSubset(metapd : pd.DataFrame, get_locs : dict = None, ign_locs : dict = None ):
  '''Gets a subset of the original dataframe
  '''
  metacopy = metapd.copy()
  if not (get_locs is None):
    for key,val in get_locs.items():
      metacopy = metacopy[metacopy[key]==val]
  if not (ign_locs is None):
    for key,val in ign_locs.items():
      metacopy = metacopy[metacopy[key]!=val]
  return metacopy

def GetAccuraciesArrays(metapd,N_obj,P_int,get_locs : dict = {},ign_locs = {}):
  np_acc = np.zeros((len(N_obj),len(P_int)))
  tf = np.zeros((len(N_obj),len(P_int),4))
  for i,n in enumerate(N_obj):
    # N_acc_high.append(GetAccuracies(metapd,
    #                                 get_locs=dict(N_objects=n),
    #                                 ign_locs=dict(target_color='blue'))[0])
    for j,p in enumerate(P_int):
      acc,truefalsethings = GetAccuracies(metapd,dict(N_distractors=n,
      P_interfere=p,**get_locs),ign_locs)
      #print(np.sum(oth))
      np_acc[i,j]=acc
      tf[i,j,:]=np.array(truefalsethings)
  return  np_acc,tf

def dprime_isright(metapd,p1='has_target',p2='isright'):
  '''
  p1: ground truth (true/false)
  p2: model was right/wrongh
  '''
  df = metapd.dropna()#subset=[p1,p2])
  hit_rate = (df[df[p1] & df[p2]].shape[0]) / (df[df[p1]].shape[0])
  false_alarm_rate = (df[~df[p1] & ~df[p2]].shape[0]) / (df[~df[p1]].shape[0])
  hit_rate = np.clip(hit_rate, 0.01, 0.99)
  false_alarm_rate = np.clip(false_alarm_rate, 0.01, 0.99)
  d_prime = norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)
  criterion = (norm.ppf(hit_rate)+norm.ppf(false_alarm_rate))/2
  return d_prime, criterion

def dprime_prob(metapd,p1='red',p2='red_probe_out'):
  '''
  p1: ground truth (true/false)
  p2: model answer (probability float, in [0.,1.])
  '''
  df = metapd[[p1,p2]]#.dropna(subset=[p1,p2])
  hit_rate = (df[df[p1].astype(bool) & (df[p2]>0.5)].shape[0]) / (df[p1].sum())
  false_alarm_rate = (df[~df[p1].astype(bool) & (df[p2]>0.5)].shape[0]) / (df[~df[p1].astype(bool)].shape[0])
  hit_rate = np.clip(hit_rate, 0.01, 0.99)
  false_alarm_rate = np.clip(false_alarm_rate, 0.01, 0.99)
  #print(hit_rate,false_alarm_rate)
  d_prime = norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)
  criterion = (norm.ppf(hit_rate)+norm.ppf(false_alarm_rate))/2
  return d_prime, criterion