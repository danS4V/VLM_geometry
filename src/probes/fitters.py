### These functions create a balanced subset of the dataset
### and train the probes
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
# import torch.nn.functional as F
import pytorch_lightning as pl
import hydra
import numpy as np

import sys
sys.path.append('../')
from vlm_datasets.utils import GetSubset
import probes.utils

def FitIndepProbeNoTest(color,shape,metapd,outputs,label,cfg):
    '''Fits probes that predict a single color-shape combination, on a balanced dataset,
    without splitting in training and test set'''
    if cfg.dataset.DATASET_TYPE== "colshape_random":
      colshape = color+shape
      
      # all keys containing the shape/color
      colorcolumns = [key for key in metapd.keys() if color in key]
      shapecolumns = [key for key in metapd.keys() if shape in key]

      # indexes of all points containing at least one figure with that shape/color
      hasshape_indexes = metapd[shapecolumns].any(axis=1)
      hascolor_indexes = metapd[colorcolumns].any(axis=1)

      # metadata of points with target shape, but different colors only
      shape_nocolor = metapd[(metapd[colshape]==0)&(hasshape_indexes)]
      # points with target color, different shapes
      noshape_color = metapd[(~hasshape_indexes)&(hascolor_indexes)].sample(len(shape_nocolor))
      
      metared = pd.concat([metapd[metapd[colshape]==1].sample(len(shape_nocolor)*2),
                          shape_nocolor,
                          noshape_color ])
      dataset = probes.utils.HasColorShapeDataset(metared,colshape,outputs)
      generator = torch.Generator(device='cuda')
    elif (cfg.dataset.DATASET_TYPE=='colshape_balanced') or (cfg.dataset.DATASET_TYPE=='colshape_unbalanced'):
      ## in these datasets, there is no asymmetry in color-shapes
      ## we just take all images with colshape as target, (balanced)
      ##              all images with a colshape but not as target
      ##                an equal amount of images with no colshape and not as target
      colshape=color+shape
      meta_target=GetSubset(metapd,get_locs={'target_color':color,'target_shape':shape})
      meta_notarget_with_colshape = GetSubset(metapd.drop(meta_target.index),
                                              get_locs={colshape:1})
      meta_notarget_nocolshape = GetSubset(metapd.drop(meta_target.index),
                                              get_locs={colshape:0})
      meta_notarget = pd.concat([meta_notarget_with_colshape,
                                 meta_notarget_nocolshape.sample(len(meta_notarget_with_colshape))])
      
      pd_train = pd.concat([meta_target,meta_notarget])
      data_train = probes.utils.HasColorShapeDataset(pd_train,
                                                    colshape,outputs)
      print('Total images to train on:',len(data_train))
      print('Images with target:', pd_train[colshape].sum())
      singletraintest = np.zeros(len(metapd))
      singletraintest[pd_train.index]=1
      generator = torch.Generator(device='cuda')
    else:
      raise KeyError(f'Dataset type unknown: {cfg.dataset.DATASET_TYPE}')
    
    print()

    dl = True if (len(data_train)%cfg.probe.batch_size)<5 else False
    train_dataloader = DataLoader(data_train, batch_size=cfg.probe.batch_size, shuffle=True,generator=generator,drop_last=dl)

    ## model and trainer
    probe = hydra.utils.instantiate(cfg.probe.probe_model,outputs.shape[2])
    probe.colshape = colshape
    
    callbacks = []
    for cb in cfg.probe.trainer_callbacks:
      callbacks.append(hydra.utils.instantiate(cb))
    logger = pl.loggers.CSVLogger(cfg.probe.log_dir, name=label+"_"+colshape,version=cfg.probe.versionlabel)
    trainer = hydra.utils.instantiate(cfg.probe.probe_trainer,callbacks=callbacks,logger=logger)
    trainer.fit(probe,train_dataloader)
    
    del data_train.data
    torch.cuda.empty_cache()
    return probe,singletraintest

