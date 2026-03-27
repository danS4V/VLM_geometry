## Evaluate templates 

import torch
import numpy as np
import pandas as pd
from ast import literal_eval
import hydra
from omegaconf import OmegaConf,DictConfig
from PIL import Image
from transformers import GenerationConfig
from transformers.utils import ModelOutput
from tqdm import tqdm
import os
import pickle
import random
import copy
from itertools import product
import torch.nn.functional as F

import sys
sys.path.append('./src/')


@torch.inference_mode()
def generate_model_output(meta: dict, toprompt: tuple, 
                          model,getstring: callable,outindexes: list,image_maker: callable,
                          cfg, constants, generation_kwargs):
  '''Generates the image and prompts the model
  Return
  ---
  str_out
  probs_out
  logits_out
  '''
  img = image_maker(meta,cfg.dataset,constants)
  prompt = str.format(cfg.task['prompt_format'], **dict(target_color=toprompt[0],target_shape = toprompt[1]))
  model_input = model.get_inputs(img,prompt)
  #with torch.inference_mode(): #decorator should do the same
  generation_output = model.model.generate(**model_input,**generation_kwargs)
  input_len = model_input["input_ids"].shape[-1]
  str_out = getstring(model,generation_output,input_len)
  outputsm = torch.nn.functional.softmax(generation_output['logits'][0],dim=-1)
  probs_out = (outputsm[0][outindexes].cpu())
  logits_out = (generation_output['logits'][0][0][outindexes].cpu())
  return str_out,probs_out,logits_out

def _generate_proj_hook_adaptive(layer_name : str, target_pos : list, 
                        prototypes: list = None):
  '''Generates a hook that performs adaptive steering: intervention magnitude scaled by 
  each token's projection onto the source concept.
  '''

  def proj_hook(model,input,output):
    if isinstance(output, tuple):
      tochange = output[0]['last_hidden_state']
    elif isinstance(output,torch.Tensor):
      tochange = output
    elif isinstance(output, ModelOutput):
      tochange = output['last_hidden_state'] 
    else:
      raise Exception(f"Hook in layer {layer_name}: output type unknown, {type(output)}")

    initshape = tochange.shape
    inittype = tochange.dtype
    tochangecloned = tochange.squeeze().to(torch.float)
    if prototypes is None:
      raise Error
    del_prot = prototypes[0]
    ins_prot = prototypes[1]
    if len(tochangecloned.shape) == 3:
      tochangecloned=tochangecloned.flatten(end_dim=1)
    
    for targettoken in range(tochangecloned.shape[0]): 
      target_proj_module = (tochangecloned[targettoken,:]@del_prot)
      tochangecloned[targettoken,:] = tochangecloned[targettoken,:]-target_proj_module*del_prot+target_proj_module*ins_prot
    
    tochange.copy_(tochangecloned.reshape(initshape).to(inittype))
    return output
  return proj_hook

def _generate_proj_hook_global(layer_name : str, target_pos : list, 
                        prototypes: list = None):
  '''Generates a hook that performs global scaling steering: intervention magnitude is 
  the same across all tokens (already baked into prototypes via template.multiplier).
  '''

  def proj_hook(model,input,output):
    if isinstance(output, tuple):
      tochange = output[0]['last_hidden_state']
    elif isinstance(output,torch.Tensor):
      tochange = output
    elif isinstance(output, ModelOutput):
      tochange = output['last_hidden_state'] 
    else:
      raise Exception(f"Hook in layer {layer_name}: output type unknown, {type(output)}")

    initshape = tochange.shape
    inittype = tochange.dtype
    tochangecloned = tochange.squeeze().to(torch.float)
    if prototypes is None:
      raise Error
    del_prot = prototypes[0]
    ins_prot = prototypes[1]
    if len(tochangecloned.shape) == 3:
      tochangecloned=tochangecloned.flatten(end_dim=1)
    
    for targettoken in range(tochangecloned.shape[0]): 
      tochangecloned[targettoken,:] = tochangecloned[targettoken,:] - del_prot + ins_prot
    
    tochange.copy_(tochangecloned.reshape(initshape).to(inittype))
    return output
  return proj_hook

@hydra.main(version_base=None, config_path='config', config_name='activation_patching')
def main(cfg: DictConfig):
  ## DATASET PARAMETERS
  N_dis = [4,16,40]    # Numbers of distractors
  P_int = [.25,.5,.75] # Proportion of conjunctive distractors (sharing one feature with object A)
  N_img = 1            

  constants = OmegaConf.to_container(cfg.dataset.constants)
  stencils = np.load('./data/stencils.npy')
  constants['SHAPES_NP'] = { 
      # arrays of greyscale (1-0) images (1 will be colored); all potential shapes to be included
      'triangle': stencils[9],
      'square': stencils[-1],
      'pentagon': stencils[59],
      'circle': stencils[-2],
      'star': stencils[-4],
      'heart': stencils[-6],
      'spade': stencils[51]
    }
  image_maker = hydra.utils.instantiate(cfg.dataset.make_image_function)

  ## Load model 
  model = hydra.utils.instantiate(cfg.model)
  yes_indexes = [model.vocab['yes'],model.vocab['Yes'],model.vocab['YES']]
  no_indexes = [model.vocab['no'],model.vocab['No'],model.vocab['NO']]
  
  templabel = cfg.template.name
  
  datapath = os.path.join('.','outputs','temp_eval',cfg.model.model_name)  
  templates = torch.load(os.path.join(datapath,f'template_{templabel}.pt')) * cfg.template.multiplier
  templates = templates.cuda()
  
  if cfg.template.type == 'token_vector':
    scaling_mode = getattr(cfg.template, 'scaling_mode', 'adaptive')  # default to adaptive for backward compat
    if scaling_mode == 'adaptive':
      hook_generator = _generate_proj_hook_adaptive
    elif scaling_mode == 'global':
      hook_generator = _generate_proj_hook_global
    else:
      raise ValueError(f"Unknown scaling_mode: {scaling_mode}. Use 'adaptive' or 'global'.")
  else:
    raise KeyError

  if 'Intern' in cfg.model.model_name:
      generation_kwargs = dict(generation_config=GenerationConfig(
      max_new_tokens= 3,
      do_sample= False, #avoid sampling outputs (greedy choice)
      return_dict_in_generate= True,  #outputs is a dictionary, following args specify its contents
      output_attentions= False, # Passed to all submodels, needed to use hooks
      output_hidden_states= False,
      output_logits= True,  # These are handy
      output_scores= False,
      ) )
      def getstring(model,output,input_len):
        return model.processor.decode(output['sequences'][0])
  else:
      generation_kwargs = { 
        "max_new_tokens": 3, 
        "do_sample": False, #avoid sampling outputs (greedy choice)
        "return_dict_in_generate": True,  #outputs is a dictionary, following args specify its contents
        "output_attentions": False, # Passed to all submodels, needed to use hooks
        "output_hidden_states": False,
        "output_logits": True,  # These are handy
        "output_scores": False,
      }
      def getstring(model,output,input_len):
          return model.processor.decode(output['sequences'][0][input_len:])

  meta_probs_t = []
  meta_logits_t = []
  meta_probs_d = []
  meta_logits_d = []
  meta_probs_trued = []
  meta_logits_trued = []
  ID = 0 #keep track of image ids
  
  colshapesdict=dict() #go from 'redsquare' to 0
  for i in range(6):
    for j in range(6):
      colshapesdict[cfg.dataset.COLORS[i]+cfg.dataset.SHAPES[j]]=i*6+j

  with tqdm(total=(len(cfg.dataset.COLORS[:])*len(cfg.dataset.SHAPES)*len(N_dis)*len(P_int)*2*N_img)) as pbar:
    for i,t_col in enumerate(cfg.dataset.COLORS.copy()[:]):
     for j,t_sha in enumerate(cfg.dataset.SHAPES.copy()[:]): #for all targets
      pbar.set_description_str(f'{t_col} {t_sha}')

      #REGISTER HOOK
      index = [0]
      tp = [] #target position for padded templates
      prototypes = [] #this one will always have [prot to delete, prot to hallucinate]
      handles  = model.register_hooks(hook_generator = hook_generator,
                              hook_layers = {'mmp': cfg.model.probe_layers['mmp']},
                              hook_generator_kwargs={'target_pos': tp, #'ind' : index,
                                                    'prototypes': prototypes,
                                                    })

      #load metadata
      metapd = pd.read_csv(os.path.join(datapath,f'{t_col}{t_sha}_metadata.csv'),
                       dtype={'ID': 'string'}, 
                      converters={'distractors': literal_eval})
      metadata = metapd.to_dict('records')
      for meta in metadata:
                #nogoodimageyet = False #changes if some answer is wrong
                ### Convention: output line is an image, [td, t, d, none] as columns
                output_probs_target = []
                output_logits_target = []
                str_out_target = []

                output_probs_dis = []
                output_logits_dis = []
                str_out_dis = []

                output_probs_truedis = []
                output_logits_truedis = []
                str_out_truedis = []
                tp.append(meta['distractors'][0]['pos'])
                distractorindex = colshapesdict[meta['dis_color']+meta['dis_shape']]
                #steer_prots.append(templates[distractorindex].to(torch.bfloat16).cuda())

                #for tp_ind,toprompt in enumerate([(t_col,t_sha),(meta['dis_color'],meta['dis_shape'])]):
                for tokeep in [True,False]: #whether to keep the target
                    meta_expendable = copy.deepcopy(meta)
                    prototypes.clear()
                    if not tokeep:
                      meta_expendable['distractors'][0]['color']=meta_expendable['dis_color']
                      meta_expendable['distractors'][0]['shape']=meta_expendable['dis_shape']
                      prototypes.append(templates[distractorindex])
                      prototypes.append(templates[6*i+j]) #remove dis, insert target
                    else:
                      prototypes.append(templates[6*i+j])
                      prototypes.append(templates[distractorindex]) #remove target, insert dis

                    str_out, probs_out, logits_out = generate_model_output(meta_expendable,
                                                                          (t_col,t_sha),
                                                                          model, getstring,yes_indexes+no_indexes,
                                                                          image_maker,cfg,constants,generation_kwargs)
                    str_out_target.append(str_out)
                    output_probs_target.append(probs_out)
                    output_logits_target.append(logits_out)

                    ## With everything, ask distractor
                    str_out, probs_out, logits_out = generate_model_output(meta_expendable,
                                                                          (meta_expendable['dis_color'],meta_expendable['dis_shape']),
                                                                          model, getstring,yes_indexes+no_indexes,
                                                                          image_maker,cfg,constants,generation_kwargs)
                    str_out_dis.append(str_out)
                    output_probs_dis.append(probs_out)
                    output_logits_dis.append(logits_out)

                    ## Ask for untouched disjunctive distractor
                    dc = meta_expendable['distractors'][-1]['color']
                    ds = meta_expendable['distractors'][-1]['shape']
                    str_out, probs_out, logits_out = generate_model_output(meta_expendable,
                                                                          (dc,ds),
                                                                          model, getstring,yes_indexes+no_indexes,
                                                                          image_maker,cfg,constants,generation_kwargs)
                    str_out_truedis.append(str_out)
                    output_probs_truedis.append(probs_out)
                    output_logits_truedis.append(logits_out)

                #keep results
                meta_probs_t.append(output_probs_target)
                meta_logits_t.append(output_logits_target)
                meta_probs_d.append(output_probs_dis)
                meta_logits_d.append(output_logits_dis)
                meta_probs_trued.append(output_probs_truedis)
                meta_logits_trued.append(output_logits_truedis)
                pbar.update(1)
                ID+=1
      # dump information 
      with open(os.path.join(datapath,f'steer_{t_col}{t_sha}_{templabel}_{cfg.template.multiplier:.2f}.pkl'),'wb') as f:
        pickle.dump(dict( probs_t = torch.stack([torch.stack(a) for a in meta_probs_t]),
                          logits_t = torch.stack([torch.stack(a) for a in meta_logits_t]),
                          probs_d = torch.stack([torch.stack(a) for a in meta_probs_d]),
                          logits_d = torch.stack([torch.stack(a) for a in meta_logits_d]),
                          probs_trued = torch.stack([torch.stack(a) for a in meta_probs_trued]),
                          logits_trued = torch.stack([torch.stack(a) for a in meta_logits_trued]),
                        ),f)
      handles['mmp'].remove()
      # clear all lists
      for l in [meta_probs_t, meta_logits_t,meta_probs_d, meta_logits_d,meta_probs_trued, meta_logits_trued]:
        l.clear()

  return

if __name__=='__main__':
  main()
