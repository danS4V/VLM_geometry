import pandas as pd
import numpy as np
from ast import literal_eval # to get dict/list from csv
import hydra
from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import ConfigAttributeError
import torch

import os
import glob
import pickle
from itertools import product

import pyrootutils, sys

import warnings
warnings.filterwarnings('ignore')

pyrootutils.setup_root('.', dotenv=True, pythonpath=False)
sys.path.append('./src')
import probes.utils
from vlm_datasets.utils import GetSubset
from probes.fitters import FitIndepProbeNoTest

def GenerateAnswerMetadata(outputdir,cfg) -> pd.DataFrame:
  '''
  Retrieves the dataframe with image properties and raw model answers
  and adds information on which color-shape combinations were 
  contained in the image.
  '''
  ## read images metadata
  all_files = sorted( glob.glob(os.path.join(outputdir , "metadata*.csv")) )
  metas = []
  for filepath in (all_files):
    metas.append(pd.read_csv(filepath, 
                          dtype={'ID': 'string'}, 
                          converters={'target_pos': literal_eval, 
                                      'distractors': literal_eval}
                ))
  metapd = pd.concat(metas,ignore_index=True).sort_values('ID')

  #record which color-shapes are there in the image
  metadata = metapd.to_dict('records')
  for meta in metadata:
    for shape in cfg.dataset.SHAPES:
      meta[shape]=0
    for color in cfg.dataset.COLORS:
      meta[color]=0
      for shape in cfg.dataset.SHAPES:
        meta[color+shape]=0 #initialize all colorshapes to zero
    if meta['has_target']:
      meta[meta['target_color']+meta['target_shape']]=1
      meta[meta['target_color']]=1
      meta[meta['target_shape']]=1
    for distractor in meta['distractors']:
      meta[distractor['color']+distractor['shape']]=1
      meta[distractor['color']]=1
      meta[distractor['shape']]=1
  
  metapd = pd.DataFrame(metadata)
  metapd.to_csv(os.path.join(outputdir,'full_metadata.csv'),index=False)
  return  metapd



@hydra.main(version_base=None, config_path='config', config_name='probe_training')
def main(cfg: DictConfig) -> None:
  outputdir = cfg.probe.outputs_path
  try:
    probelabel = cfg.layer
  except ConfigAttributeError:
    probelabel = input('Insert a layer label: ')
  probeoutputdir = os.path.join(outputdir,probelabel+"_probes")
  if not os.path.exists(probeoutputdir):
    os.makedirs(probeoutputdir)

  ## Create useful metadata
  print('Generating metadata')
  metapd = GenerateAnswerMetadata(outputdir, cfg)

  ## create a dictionary to keep track of which images were used in training
  ## traintest[colorshape] i a tensor: 0=not used, 1=training, 2=kept apart for testing, balanced 
  ## test split is not supposed to be used as validation (anymore)
  ## if training only some probes with argument select_colorshape, keeps old info
  traintestpath = os.path.join(probeoutputdir,f'traintest_{cfg.probe.versionlabel}.pkl')
  if os.path.exists(traintestpath):
    with open(traintestpath,'rb') as f:
      traintest= pickle.load(f)
  else:
    traintest = {}

  ## load activations
  print('Loading activations')
  tensorpaths = sorted( glob.glob(os.path.join(outputdir,f"{probelabel}*.*")) )
  outputs = []
  if ".pt" in tensorpaths[0]:
    for i,path in enumerate(tensorpaths):
      print(path)
      outputs.append(torch.load(path,map_location='cpu'))
    outputs = torch.cat(outputs,dim=0)
  elif ".pkl" in tensorpaths[0]:
    for i,path in enumerate(tensorpaths):
      print(path)
      with open(path,'rb') as f:
        outputs+=pickle.load(f)
    # use only part of the tokens
    if cfg.model.probe_train_indexes[probelabel] is not None:
      indexes = cfg.model.probe_train_indexes[probelabel]
      print('Cutting hidden layers...',indexes)
      for i,hiddenout in enumerate(outputs):
        outputs[i] = hiddenout[0,indexes[0]:indexes[1],:]
    outputs= torch.stack(outputs).squeeze().to(torch.float32)
    print(f"hidden layer shape: {outputs.shape}")

  torch.set_default_device('cuda')

  if cfg.probe.type == 'independent':
    if cfg.select_colshapes is not None:
      cslist = cfg.select_colshapes
    else:
      cslist = product(cfg.dataset.COLORS,cfg.dataset.SHAPES)
    for color,shape in cslist:
      print("\033[92m",color,shape,"\033[0m")
      probe, singletraintest = FitIndepProbeNoTest(color,shape,metapd,outputs,probelabel,cfg)
      torch.save(probe.state_dict(),os.path.join(probeoutputdir,color+shape+f'_{cfg.probe.versionlabel}.nn'))
      del probe
      traintest[color+shape]=singletraintest
      with open(traintestpath,'wb') as f:
        pickle.dump(traintest,f)
        #sometimes i may want to ctrl-c training, but keep info on what has already be done;
        #so i overwrite it every time instead then just at the end
      torch.cuda.empty_cache()
    return
  else: 
    raise NotImplementedError(f"Probe type {cfg.probe.type} not implemented")



if __name__== "__main__":
  main()
