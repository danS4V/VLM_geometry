import torch
import numpy as np
import hydra
from omegaconf import OmegaConf,DictConfig
from PIL import Image
from transformers import GenerationConfig
from transformers.utils import ModelOutput
from tqdm import tqdm
import os
import pickle
import gc

import sys
sys.path.append('./src/')

def _generate_output_hook(layer_name : str, current_colshape: list, inner_output : dict):
  '''Returns an hook that saves layer outputs to the list inner_output[*colshape*]
  '''
  def prompt_output_hook(model, input, output):
      if isinstance(output, tuple):
        output = output[0] 
        tosave = output['last_hidden_state'].detach().cpu() 
      elif isinstance(output,torch.Tensor):
        tosave = output.detach().cpu()
      elif isinstance(output, ModelOutput):
        tosave = output['last_hidden_state'].detach().cpu() 
      else:
        raise Exception(f"Hook in layer {layer_name}: output type unknown, {type(output)}")

      # print(tosave.shape[1]) 
      if tosave.shape[1]>1: # whole input, next batch index
        inner_output[current_colshape[-1]].append(tosave)
      # else, it's the generation outputs, which we don't need.
  return prompt_output_hook

@hydra.main(version_base=None, config_path='config', config_name='activation_patching')
def main(cfg: DictConfig):
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

  ## Create array of positions
  p0 = (224,224) #central position
  d=28 #patch size
  delta1 = [(0,d),(0,-d),(d,0),(-d,0)]
  delta2 = [(d,d),(d,-d),(-d,-d),(-d,d)]

  pos_cent = [p0,]+ [(p0[0]+4*dx,p0[1]+4*dy) for dx, dy in delta1]+[(p0[0]+7*dx,p0[1]+7*dy) for dx, dy in delta2]
  pos = []
  pos +=pos_cent
  dd = 5 #
  displacements = [(0,dd),(0,-dd),(dd,0),(-dd,0),(dd,dd),(dd,-dd),(-dd,-dd),(-dd,dd)]
  for dx,dy in displacements:
    pos += [(px+dx,py+dy) for px,py in pos_cent]
  
  templatedir = os.path.join(".","outputs",cfg.model.model_name,
                             cfg.task.task_type,'data_templates_81')
  if not os.path.exists(templatedir):
    os.makedirs(templatedir)

  ## Load model and register hook
  model = hydra.utils.instantiate(cfg.model)
  yes_indexes = [model.vocab['yes'],model.vocab['Yes'],model.vocab['YES']]
  no_indexes = [model.vocab['no'],model.vocab['No'],model.vocab['NO']]
  
  if "InternVL3" in cfg.model.model_name:
      generation_kwargs = { 
            "max_new_tokens": 3, 
            "do_sample": False, #avoid sampling outputs (greedy choice)
            "return_dict_in_generate": True,  #outputs is a dictionary, following args specify its contents
            "output_attentions": False, # Passed to all submodels, needed to use hooks
            "output_hidden_states": False,
            "output_logits": True,  # These are handy
            "output_scores": False,
            "pad_token_id": 151645,
          }
      def getstring(model,output,input_len):
            return model.processor.decode(output['sequences'][0][input_len:])
  elif 'Intern' in cfg.model.model_name:
      generation_kwargs = dict(generation_config=GenerationConfig( 
      max_new_tokens= 3, 
      do_sample= False, #avoid sampling outputs (greedy choice)
      return_dict_in_generate= True,  #outputs is a dictionary, following args specify its contents
      output_attentions= False, # Passed to all submodels, needed to use hooks
      output_hidden_states= False,
      output_logits= True,  # These are handy
      output_scores= False,
      ) )
      # I'll move this to the model class sooner or later
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

  activations = {}
  current_colshape = []
  handles  = model.register_hooks(hook_generator = _generate_output_hook,
                          hook_layers = {'mmp': cfg.model.probe_layers['mmp']},
                          hook_generator_kwargs={'inner_output' : activations,
                                                 'current_colshape': current_colshape})
  activations_last = {}
  handles_last  = model.register_hooks(hook_generator = _generate_output_hook,
                          hook_layers = {'last': cfg.model.probe_layers['last_mlp']},
                          hook_generator_kwargs={'inner_output' : activations_last,
                                                 'current_colshape': current_colshape})
  output_probs = {}

  ## Collect empty image activations
  activations['empty']=[]
  activations_last['empty']=[]
  current_colshape.append('empty')
  meta = dict(distractors=[])
  img = image_maker(meta,cfg.dataset,constants)
  target_properties = dict(target_color='red', #just as a placeholder, activations are not affected
              target_shape = 'square')
  prompt = str.format(cfg.task['prompt_format'],**target_properties)
  model_input = model.get_inputs(img,prompt)
  with torch.inference_mode():
    generation_output = model.model.generate(**model_input,**generation_kwargs)


  with tqdm(total=36*(len(pos))) as pbar:
    for col in cfg.dataset.COLORS:
      for sha in cfg.dataset.SHAPES:
        current_colshape.append(col+sha)
        activations[col+sha] = []
        activations_last[col+sha] = []
        output_probs[col+sha] = []
        for i in range(len(pos)):
          meta = dict(distractors=[dict(color=col,shape=sha,pos=pos[i])])
          img = image_maker(meta,cfg.dataset,constants)
          target_properties = dict(target_color=col,
                      target_shape = sha,
                      pos = pos[i])
          prompt = str.format(cfg.task['prompt_format'],**target_properties)
          model_input = model.get_inputs(img,prompt)
          with torch.inference_mode():
            generation_output = model.model.generate(**model_input,**generation_kwargs)
          outputsm = torch.nn.functional.softmax(generation_output['logits'][0],dim=-1)
          output_probs[col+sha].append(outputsm[0][yes_indexes+no_indexes].cpu())
          pbar.update(1)
      torch._dynamo.reset()
      gc.collect()
      torch.cuda.empty_cache()
  

  with open(os.path.join(templatedir,'activations_mmp.pkl'),'wb') as f:
    pickle.dump(activations,f)
  with open(os.path.join(templatedir,'activations_last.pkl'),'wb') as f:
    pickle.dump(activations_last,f)
  with open(os.path.join(templatedir,'output_probs.pkl'),'wb') as f:
    pickle.dump(output_probs,f)

  with open(os.path.join(templatedir,'pos.txt'),'wt') as f:
    print(pos,file=f)



  return

if __name__=='__main__':
  main()
