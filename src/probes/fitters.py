### These functions create a balanced subset of the dataset
### and train the probes
import functools
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
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

def FitCorSProbeNoTest(proptype,prop,metapd,outputs,label,cfg):
  # dataset is already close to balanced (~45% without prop)
  #dataset = probes.utils.HasColorShapeDataset(metapd,prop,outputs)
  generator = torch.Generator(device='cuda')

  #pd_train = metapd.sample(frac=0.7)
  data_train = probes.utils.HasColorShapeDataset(metapd,
                                                    prop,outputs)
  # pd_test = metapd.drop(pd_train.index)
  # data_test = probes.utils.HasColorShapeDataset(pd_test,
  #                                                 prop,outputs)
  singletraintest = np.ones(len(metapd))
  # singletraintest[pd_train.index]=1
  # singletraintest[pd_test.index]=2

  dl = True if (len(data_train)%cfg.probe.batch_size)<5 else False
  train_dataloader = DataLoader(data_train, batch_size=cfg.probe.batch_size, shuffle=True,generator=generator,drop_last=dl)
  #test_dataloader = DataLoader(data_test, batch_size=len(data_test),generator=generator)

  ## model and trainer
  probe = hydra.utils.instantiate(cfg.probe.probe_model,outputs.shape[2])
  probe.colshape = prop
  
  callbacks = []
  for cb in cfg.probe.trainer_callbacks:
    callbacks.append(hydra.utils.instantiate(cb))
  logger = pl.loggers.CSVLogger(cfg.probe.log_dir, name=label+"_"+prop,version=cfg.probe.versionlabel)
  trainer = hydra.utils.instantiate(cfg.probe.probe_trainer,callbacks=callbacks,logger=logger)
  trainer.fit(probe,train_dataloader)#,test_dataloader)
  
  #del data_test.data
  del data_train.data
  torch.cuda.empty_cache()
  return probe,singletraintest

def FitJointProbeNoTest(colshapes, metapd, outputs, label, cfg):
    '''Fits a single joint probe predicting all K color-shape concepts simultaneously.

    Uses all images without per-concept balancing. Per-concept pos_weight
    (neg_count / pos_count) compensates for class imbalance.

    Parameters
    ----------
    colshapes : list[str]
        Ordered list of K concept strings (e.g. ['redsquare', 'redcircle', ...]).
        Order must be preserved and saved alongside the probe weights.
    metapd : pd.DataFrame
        Full metadata with one binary column per concept in colshapes.
    outputs : torch.Tensor
        Activations tensor of shape (N, T, D).
    label : str
        Layer label used for logging.
    cfg : DictConfig
        Hydra config (must have cfg.probe.* entries from attn_joint.yaml).

    Returns
    -------
    probe : LAttnProbeJoint
    '''
    dataset = probes.utils.HasColorShapeMultiLabelDataset(metapd, colshapes, outputs)
    print('Total images for joint training:', len(dataset))

    # Per-concept pos_weight = neg_count / pos_count to handle imbalance
    labels    = dataset.labels          # (N, K) on CPU
    pos_count = labels.sum(dim=0).clamp(min=1)
    neg_count = (len(labels) - labels.sum(dim=0)).clamp(min=1)
    pos_weight = (neg_count / pos_count).cuda()

    dl_drop = (len(dataset) % cfg.probe.batch_size) < 5
    train_dataloader = DataLoader(
        dataset, batch_size=cfg.probe.batch_size, shuffle=True,
        generator=torch.Generator(device='cuda'), drop_last=dl_drop,
    )

    probe = hydra.utils.instantiate(cfg.probe.probe_model, outputs.shape[2], len(colshapes))
    # Override loss to inject pos_weight (data-dependent, cannot live in YAML)
    probe.Loss = functools.partial(
        F.binary_cross_entropy_with_logits, pos_weight=pos_weight
    )

    callbacks = []
    for cb in cfg.probe.trainer_callbacks:
        callbacks.append(hydra.utils.instantiate(cb))
    logger = pl.loggers.CSVLogger(
        cfg.probe.log_dir, name=label + '_joint', version=cfg.probe.versionlabel
    )
    trainer = hydra.utils.instantiate(cfg.probe.probe_trainer, callbacks=callbacks, logger=logger)
    trainer.fit(probe, train_dataloader)

    del dataset.data
    torch.cuda.empty_cache()
    return probe
