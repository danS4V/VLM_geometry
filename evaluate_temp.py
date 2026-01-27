## Evaluate templates 

import torch
import numpy as np
import pandas as pd
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

import sys
sys.path.append('./src/')
from vlm_datasets.shapecol_unbal import generate_positions

def generate_meta(cfg : DictConfig,constants,t_col,t_sha,N,P,pref,imgnum,ID):
  '''
  Modified to perform steering experiments only.
  Return
  meta: dict
    metadata for one image with desired properties.'''
  dis_shapes = OmegaConf.to_container(cfg.dataset.SHAPES).copy()
  dis_colors = OmegaConf.to_container(cfg.dataset.COLORS).copy()
  dis_shapes.remove(t_sha)
  dis_colors.remove(t_col)
  d_col = random.choice(dis_colors)
  d_sha = random.choice(dis_shapes)

  pos = generate_positions(N+1,constants['CANVAS_SIZE'],constants['STENCIL_SIZE'])

  meta=dict(
    ID = str(ID).zfill(4),  # Incremental number padded to 4 digits
    target_color = t_col,
    target_shape = t_sha,
    pref = pref,
    N_distractors = N,
    P_interfere = P,
    dis_color = d_col,
    dis_shape = d_sha,
    distractors = [dict(color=t_col,shape=t_sha,pos=pos[0].tolist()),]
                  #dict(color=d_col,shape=d_sha,pos=pos[1].tolist())] this one is not included
  )
  ## add conjunctive distractors
  #assert (N*P)%1==0
  N_conj_dis = int(N*P)
  for i in range(N_conj_dis):
    if pref=='col':
      meta['distractors'].append(dict(color=t_col,
                                              shape=random.choice(dis_shapes),
                                              pos=pos[i+1].tolist())) #0&1 are for target and d*
    elif pref=='sha':
      meta['distractors'].append(dict(color=random.choice(dis_colors),
                                              shape=t_sha,
                                              pos=pos[i+1].tolist())) #0&1 are for target and d*
  N_disj_dis = N-N_conj_dis
  dis_shapes.remove(d_sha)
  dis_colors.remove(d_col)
  for i in range(N_conj_dis,N_conj_dis+N_disj_dis):
    meta['distractors'].append(dict(color=random.choice(dis_colors),
                                            shape=random.choice(dis_shapes),
                                            pos=pos[i+1].tolist()))
  return meta

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
  #with torch.inference_mode(): 
  generation_output = model.model.generate(**model_input,**generation_kwargs)
  input_len = model_input["input_ids"].shape[-1]
  str_out = getstring(model,generation_output,input_len)
  outputsm = torch.nn.functional.softmax(generation_output['logits'][0],dim=-1)
  probs_out = (outputsm[0][outindexes].cpu())
  logits_out = (generation_output['logits'][0][0][outindexes].cpu())
  return str_out,probs_out,logits_out



@hydra.main(version_base=None, config_path='config', config_name='activation_patching')
def main(cfg: DictConfig):
  ## DATASET PARAMETERS
  N_dis = [4,16,40]    # Numbers of distractors
  P_int = [.25,.5,.75] # Proportion of conjunctive distractors
  N_img = 1            # Number of images for each configurations (must be still multiplied by 4)


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

  if 'Intern' in cfg.model.model_name:
      generation_kwargs = dict(generation_config=GenerationConfig( 
      max_new_tokens= 3, 
      do_sample= False, #avoid sampling outputs (greedy choice)
      return_dict_in_generate= True,  #outputs is a dictionary, following args specify its contents
      output_attentions= False, # Passed to all submodels, needed to use hooks
      output_hidden_states= False,
      output_logits= True,
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
        "output_logits": True,
        "output_scores": False,
      }
      def getstring(model,output,input_len):
          return model.processor.decode(output['sequences'][0][input_len:])

  metadata = []
  meta_probs_t = []
  meta_logits_t = []
  meta_probs_d = []
  meta_logits_d = []
  meta_probs_trued = []
  meta_logits_trued = []
  ID = 0 #keep track of image ids
  skipped_counter = 0
  datapath = os.path.join('.','outputs','temp_eval',cfg.model.model_name)
  if not os.path.exists(datapath):
    os.makedirs(datapath)
  with tqdm(total=len(cfg.dataset.COLORS[3:])*len(cfg.dataset.SHAPES)*(len(N_dis)*len(P_int)*2*N_img)) as pbar:
    for t_col in cfg.dataset.COLORS.copy()[3:]:
     for t_sha in cfg.dataset.SHAPES.copy(): #for all targets
      pbar.set_description_str(f'{skipped_counter:3.0f} images skipped, {t_col} {t_sha}')
      for N in N_dis:
       for P in P_int:
         for pref in ['col','sha']:
          for imgnum in range(N_img): 
            nogoodimageyet = True
            ntry = 0
            while nogoodimageyet: 
                ### LOOP LOGIC: try 10 times (ntry<10) to get all right answers, otherwise
                ## give up and keep what we have
                nogoodimageyet = False #changes if some answer is wrong
                # Select first disjunctive distractor
                meta = generate_meta(cfg,constants,t_col,t_sha,N,P,pref,imgnum,ID)

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

                #leave target, ask both, then swap target-distractor, ask both again

                for tokeep in [True,False]: #whether to keep the target, or use the disjunctive distractor
                    meta_expendable = copy.deepcopy(meta)
                    if not tokeep:
                      meta_expendable['distractors'][0]['color']=meta_expendable['dis_color']
                      meta_expendable['distractors'][0]['shape']=meta_expendable['dis_shape']

                    ## Ask target
                    str_out_t, probs_out_t, logits_out_t = generate_model_output(meta_expendable,
                                                                          (t_col,t_sha),
                                                                          model, getstring,yes_indexes+no_indexes,
                                                                          image_maker,cfg,constants,generation_kwargs)
                    if (('yes' in str_out_t or 
                        'Yes' in str_out_t or 
                        'YES' in str_out_t) != tokeep or
                        ('no'  in str_out_t or 'No'  in str_out_t or 
                         'NO'  in str_out_t) == tokeep
                        ) and ntry<10: #if answer is yes and target wasn't there, or answ was no and target was there
                      skipped_counter+=1
                      ntry+=1
                      pbar.set_description_str(f'{skipped_counter:3.0f} images skipped, {t_col} {t_sha}')
                      nogoodimageyet=True
                      break #yes probs failed, generate another image.
                    str_out_target.append(str_out_t)
                    output_probs_target.append(probs_out_t)
                    output_logits_target.append(logits_out_t)

                    ## Ask distractor
                    str_out, probs_out, logits_out = generate_model_output(meta_expendable,
                                                                          (meta_expendable['dis_color'],meta_expendable['dis_shape']),
                                                                          model, getstring,yes_indexes+no_indexes,
                                                                          image_maker,cfg,constants,generation_kwargs)
                    if (('yes' in str_out or 
                        'Yes' in str_out or 
                        'YES' in str_out) == tokeep or
                        ('no'  in str_out or 'No'  in str_out or 
                         'NO'  in str_out) != tokeep
                        ) and ntry<10:
                      #if (probs_out[:3].sum()>probs_out[3:])==tokeep:
                      skipped_counter+=1
                      ntry+=1
                      pbar.set_description_str(f'{skipped_counter:3.0f} images skipped, {t_col} {t_sha}')
                      nogoodimageyet = True
                      break #yes probs failed, generate another image.
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
                    if (not ('yes' in str_out or 
                        'Yes' in str_out or 
                        'YES' in str_out)) and ntry<10:
                      skipped_counter+=1
                      ntry+=1
                      pbar.set_description_str(f'{skipped_counter:3.0f} images skipped, {t_col} {t_sha}')
                      nogoodimageyet = True
                      break #yes probs failed, generate another image.
                    str_out_truedis.append(str_out)
                    output_probs_truedis.append(probs_out)
                    output_logits_truedis.append(logits_out)
            #keep results
            metadata.append(meta)
            meta_probs_t.append(output_probs_target)
            meta_logits_t.append(output_logits_target)
            meta_probs_d.append(output_probs_dis)
            meta_logits_d.append(output_logits_dis)
            meta_probs_trued.append(output_probs_truedis)
            meta_logits_trued.append(output_logits_truedis)

            pbar.update(1)
            ID+=1
      # dump information each time one target is completed
      metapd = pd.DataFrame(metadata)
      metapd.to_csv(os.path.join(datapath,f'{t_col}{t_sha}_metadata.csv'),index=False)
      with open(os.path.join(datapath,f'{t_col}{t_sha}_clean.pkl'),'wb') as f:
        pickle.dump(dict( probs_t = torch.stack([torch.stack(a) for a in meta_probs_t]),
                          logits_t = torch.stack([torch.stack(a) for a in meta_logits_t]),
                          probs_d = torch.stack([torch.stack(a) for a in meta_probs_d]),
                          logits_d = torch.stack([torch.stack(a) for a in meta_logits_d]),
                          probs_trued = torch.stack([torch.stack(a) for a in meta_probs_trued]),
                          logits_trued = torch.stack([torch.stack(a) for a in meta_logits_trued]),
                        ),f)
      # clear all lists
      for l in [metadata,  meta_probs_t, meta_logits_t,meta_probs_d, meta_logits_d,meta_probs_trued, meta_logits_trued]:
        l.clear()

  return

if __name__=='__main__':
  main()
