import pandas as pd
from ast import literal_eval # to get dict/list from csv
import hydra
from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import ConfigAttributeError
import torch
import os
from transformers import GenerationConfig
from transformers.utils import ModelOutput, logging
from tqdm import tqdm
import pickle
#import gc #garbage collection, but I should have removed all reference cycles already

import pyrootutils, sys

def _generate_output_hook(layer_name : str,inner_output : dict):
  '''Returns an hook that saves layer outputs to the list inner_output[layer_name],
  ignoring the outputs during generation.
  '''
  inner_output[layer_name] = []# torch.empty(0)
  def prompt_output_hook(model, input, output):
      if isinstance(output, tuple):
        output = output[0] #usually other indexes are attentions
        tosave = output['last_hidden_state'].detach().cpu() # without .cpu (even with .clone) raises error
      elif isinstance(output,torch.Tensor):
        tosave = output.detach().cpu()
      elif isinstance(output, ModelOutput):
        tosave = output['last_hidden_state'].detach().cpu() # without .cpu (even with .clone) raises error
      # 'last_hidden_state' should be model-agnostic, but im not sure
      else:
        raise Exception(f"Hook in layer {layer_name}: output type unknown, {type(output)}")

      # print(tosave.shape[1]) this is enough to break dynamo
      if tosave.shape[1]>1: # whole input, next batch index
        inner_output[layer_name].append(tosave)
        #torch.cat([inner_output[layer_name].detach().clone(), tosave], 
        #           dim=0, out=inner_output[layer_name])
      # else, it's the generation outputs, which we don't need.
  return prompt_output_hook

@hydra.main(version_base=None, config_path='config', config_name='activations_gen')
def main(cfg: DictConfig) -> None:
  ## load metadata
  datadir = os.path.join(".","data",cfg.dataset.DATASET_NAME)
  metadata = pd.read_csv(os.path.join(datadir,'metadata.csv'), 
                        dtype={'ID': 'string'}, 
                        converters={'target_pos': literal_eval, 
                                    'distractors': literal_eval}
            ).to_dict('records')

  ## load model and register hooks
  model = hydra.utils.instantiate(cfg.model)
  yes_indexes = [model.vocab['yes'],model.vocab['Yes'],model.vocab['YES']]
  no_indexes = [model.vocab['no'],model.vocab['No'],model.vocab['NO']]
  inner_out = {}
  if cfg.model.probe_layers is not None:
    handles  = model.register_hooks(hook_generator = _generate_output_hook,
                        hook_layers = cfg.model.probe_layers,
                        hook_generator_kwargs={'inner_output': inner_out})

  ## run the model
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

  # path format: outputs/*model_name*/*task_type*/*dataset_name*
  # append batch number
  # same task type may save different activations at different times.
  output_dir = os.path.join(".","outputs",cfg.model.model_name,cfg.task.task_type,cfg.dataset.DATASET_NAME)
  print(f'Saving data at {output_dir}')
  if not os.path.exists(output_dir):
    os.makedirs(output_dir) #recursively creates all needed directories

  ## run in batches
  if not cfg.test:
    batch_range = range(0,len(metadata), cfg.task.batch_size)
  else:
    batch_range = [len(metadata)-3,]

  if cfg.task.skip_index is not None:
    batch_skip = cfg.task.skip_index
  else:
    batch_skip=0
  if cfg.task.stop_index is not None:
    batch_stop = cfg.task.stop_index
  else:
    batch_stop= len(batch_range)

  for batch_start in batch_range[batch_skip:batch_stop]:
    batch_num = int(batch_start/cfg.task.batch_size)
    print(f"Processing batch {batch_num+1} of {len(batch_range)}")

    for key in inner_out:
      inner_out[key] = []
    logits = []
    output_probs = []

    for meta in tqdm(metadata[batch_start:batch_start+cfg.task.batch_size],desc='Running inference'):
      #torch.compiler.cudagraph_mark_step_begin()
      img_path = os.path.join(datadir,f'img/{meta['ID']}.png')
      prompt = str.format(cfg.task['prompt_format'],**meta)
      model_input = model.get_inputs(img_path,prompt)
      with torch.inference_mode():
        generation_output = model.model.generate(**model_input,**generation_kwargs)
      logits.append(generation_output['logits'][0][0][yes_indexes+no_indexes].detach().cpu()) # batch 0, token 0 of the output sequence
      input_len = model_input["input_ids"].shape[-1]
      str_out = getstring(model,generation_output,input_len)
      meta['str_out'] = str_out
      if meta['has_target']:
        meta['isright'] = 'yes' in meta['str_out'] or 'Yes' in meta['str_out'] or 'YES' in meta['str_out']
      else:
        meta['isright'] = 'no'  in meta['str_out'] or 'No'  in meta['str_out'] or 'NO'  in meta['str_out']
      outputsm = torch.nn.functional.softmax(generation_output['logits'][0],dim=-1)
      output_probs.append(outputsm[0][yes_indexes+no_indexes].cpu())
      meta['input_tokens']=input_len


    ## Print some output to check everything is working
    if cfg.test:
      print('Tensors shapes:')
      for key in inner_out:
        print("  ",key,":",torch.cat(inner_out[key],dim=0).shape if inner_out[key][0].shape[0]==1 else torch.stack(inner_out[key]).shape)
      #print("  logits :",torch.stack(logits).shape)
      print("yes/no probs:",torch.stack(output_probs).shape)
      print(f'String outputs : \n\t{metadata[-2]['str_out']}\n\t{metadata[-1]['str_out']}')
      #sys.exit()

    ## save data
    # save hidden states and logits as pytorch tensors
    for key in inner_out:
      with open(os.path.join(output_dir,f"{key}{batch_num}.pkl"),'wb') as f:
        pickle.dump(inner_out[key],file=f)
      ## torch tensors must have well-defined shapes, but pent_agon is made of two tokens and breaks it all:(
      # if inner_out[key][0].shape[0]==1:
      #   # If a useless index is already there, use it to concatenate tensors
      #   torch.save(torch.cat(inner_out[key],dim=0),os.path.join(output_dir,f"{key}{batch_num}.pt"))
      # else:
      #   torch.save(torch.stack(inner_out[key]),os.path.join(output_dir,f"{key}{batch_num}.pt"))

    torch.save(torch.stack(logits), os.path.join(output_dir,f"logits{batch_num}.pt"))  
    torch.save(torch.stack(output_probs), os.path.join(output_dir,f"output_probs{batch_num}.pt"))
    # save metadata (everything, redundant)
    pd.DataFrame(metadata[batch_start:batch_start+cfg.task.batch_size]).to_csv(os.path.join(output_dir,f'metadata{batch_num}.csv'),index=False)
    #print('Done')



if __name__== "__main__":
  pyrootutils.setup_root('.', dotenv=True, pythonpath=False)
  sys.path.append('./src')
  logging.set_verbosity('ERROR')
  main()