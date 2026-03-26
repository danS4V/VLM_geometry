### Generates a csv with all probe outputs and saves attention logits
## needs probes to be already trained on a dataset, and another 
## dataset named "*datasetname*_test" with all activations


import pandas as pd
import numpy as np
from ast import literal_eval # to get dict/list from csv
import hydra
from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import ConfigAttributeError
import torch
from tqdm import tqdm
from itertools import product

import os
import glob
import pickle

import pyrootutils, sys

import warnings
warnings.filterwarnings('ignore')

pyrootutils.setup_root('.', dotenv=True, pythonpath=False)
sys.path.append('./src')
import probes.utils
from vlm_datasets.utils import GetSubset

@hydra.main(version_base=None, config_path='config', config_name='probe_training')
def main(cfg: DictConfig) -> None:
  # Run probes over a dataset, store attentions and outputs

  ### Load dataset metadata
  try:
    layerlabel = cfg.layer
  except ConfigAttributeError:
    layerlabel = input('Insert a layer label: ')
  
  testoutputdir = os.path.join(cfg.output_root,cfg.model.model_name,cfg.task.task_type,
                           cfg.dataset.DATASET_NAME+'_test')
  ## probes are those trained on the version with no 'test'
  probeoutputdir = os.path.join(cfg.output_root,cfg.model.model_name,cfg.task.task_type,
                                cfg.dataset.DATASET_NAME,layerlabel+"_probes")
  testprobeoutputdir = os.path.join(testoutputdir,layerlabel+"_probes")
  

  if os.path.exists(os.path.join(testoutputdir,'full_metadata.csv')):
    print('Loading existing metadata')
    metapd = pd.read_csv(os.path.join(testoutputdir,'full_metadata.csv'), 
                            dtype={'ID': 'string'}, 
                            converters={'target_pos': literal_eval, 
                                        'distractors': literal_eval}
                )
  else:
    print('Generating metadata')
    from train_probes import GenerateAnswerMetadata
    metapd = GenerateAnswerMetadata(testoutputdir, cfg)

  ### TEST SET EVALUATION
  print('Processing test set')
  evaluate(metapd, cfg,layerlabel,probeoutputdir,testprobeoutputdir,testoutputdir)

  ### TRAINING SET EVALUATION
  print('Processing training set')
  testoutputdir = testoutputdir.replace('_test','')
  if os.path.exists(os.path.join(testoutputdir,'full_metadata.csv')):
    print('Loading existing metadata')
    metapd = pd.read_csv(os.path.join(testoutputdir,'full_metadata.csv'), 
                            dtype={'ID': 'string'}, 
                            converters={'target_pos': literal_eval, 
                                        'distractors': literal_eval}
                )
  else:
    print('Generating metadata')
    from train_probes import GenerateAnswerMetadata
    metapd = GenerateAnswerMetadata(testoutputdir, cfg)
  evaluate(metapd,cfg,layerlabel,probeoutputdir,
           testprobeoutputdir.replace('_test',''),testoutputdir)



def evaluate(metapd,cfg,layerlabel,probeoutputdir,testprobeoutputdir,testoutputdir):
  '''
  Parameters:
  -----------
  metapd : pd.DataFrame
  cfg : Config
  probeoutputdir : str
    directory where probes are stored
  testprobeoutputdir : str
    directory where activations to use are stored, and where outputs will be saved
  testoutputdir : str
    directory where the activations are stored
  '''

  versionlabel= cfg.probe.versionlabel 

  ### Load activations
  outputs =[]
  tensorpaths = sorted( glob.glob(os.path.join(testoutputdir,layerlabel+"*.*")) )

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
    if cfg.model.probe_train_indexes[layerlabel] is not None:
      indexes = cfg.model.probe_train_indexes[layerlabel]
      print('Cutting hidden layers...',indexes)
      for i,hiddenout in enumerate(outputs):
        outputs[i] = hiddenout[0,indexes[0]:indexes[1],:]
    outputs= torch.stack(outputs).squeeze().to(torch.float32)

  ##################
  ### RUN PROBES ###
  probe = hydra.utils.instantiate(cfg.probe.probe_model,in_dim=outputs.shape[2])

  attns={}
  probeout={}
  for color in cfg.dataset.COLORS[:]:
    for shape in cfg.dataset.SHAPES[:]:
      attns[color+shape]=torch.empty(0)
      probeout[color+shape]=torch.empty(0)
  colshapes = cfg.dataset.COLORS[:].copy()+(cfg.dataset.SHAPES.copy())
  for prop in colshapes:
    attns[prop]=torch.empty(0)
    probeout[prop]=torch.empty(0)

  for startindex in tqdm(range(0,len(metapd),1000)):

  #conjunctive ones
    cudoutputs = outputs[startindex:startindex+1000].to('cuda',dtype=torch.float32)
    for color in tqdm(cfg.dataset.COLORS[:],desc='Color',leave=False):
      for shape in tqdm(cfg.dataset.SHAPES[:],desc='Shape',leave=False):
        colshape = color+shape
        probe.load_state_dict(torch.load(
          os.path.join(probeoutputdir,colshape+f'_{versionlabel}.nn')
        ))

        probe.eval()
        with torch.no_grad():
          logits, attn = probe.cuda()(cudoutputs,output_attn_logits=True)
          probeout[colshape]= torch.cat( (probeout[colshape], torch.nn.functional.sigmoid(logits).detach().cpu()) )
          attns[colshape]=torch.cat([attns[colshape],attn.squeeze().cpu()])
        del logits
        del attn
    del cudoutputs
    torch.cuda.empty_cache()

  ##LOOP ON ALL PROBES
  for color in cfg.dataset.COLORS[:]:
    for shape in cfg.dataset.SHAPES[:]:
      metapd[color+shape+'_probe_out']=probeout[color+shape]
 
  ### Save stuff
  if not os.path.exists(testprobeoutputdir):
    os.makedirs(testprobeoutputdir)
  metapd.to_csv(os.path.join(testprobeoutputdir,f'full_metadata_{versionlabel}.csv'),index=False)
  with open(os.path.join(testprobeoutputdir,f'attns_logits_{versionlabel}.pkl'),'wb') as f:
    pickle.dump(attns,f)


if __name__ == '__main__':
  main()
