import torch
from models.model import Model
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

class QwenModel(Model):
  '''Class to store Qwen 2.5 model, processor and vocab.
  '''

  def __init__(self,model_id = 'Qwen/Qwen2.5-VL-7B-Instruct',**kwargs) -> None:
    self.__dict__.update(kwargs)
    self.model_id = model_id
    self.model, self.processor, self.vocab = self._get_qwen(model_id)

  def _get_qwen(self,model_id):
    ''' Retrieves model, processor and tokenizer vocab from HuggingFace.

    Parameters
    ----------
    model_id : str
      Defaults to google/gemma-3-12b-it.

    Returns
    -------
    gemma_model :
    gemma_processor :
    gemma_vocab : 
    '''

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,torch_dtype='auto', 
            device_map='auto',
            attn_implementation = 'eager' # needed to use output_attention
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    vocab = processor.tokenizer.get_vocab()
    return model,processor, vocab

  ###############
  ## Inference ##
  ###############

  def get_inputs(self,img_path,query_string):
    '''Applies the processor and adds system message to the query.
    '''
  
    messages = [
      {
        "role": "user",
        "content": [{"type": "image", "image": img_path},
                    {"type": "text", "text": query_string}]
      }]
    inputs = self.processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(self.model.device, dtype=torch.bfloat16)
    return inputs

  def ask(self,img_path,query) -> dict:
    '''Runs the model on given input (image and query).

    Parameters
    ----------
    img_path : str
      Path to an image (is None possible?)
    query : str
      User prompt

    Returns
    -------
    dict
      Dictionary containing the query, output and first token logits ('Q', 'A', 'scores').
    '''
    inputs = self.get_inputs(img_path,query)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generation = self.model.generate(**inputs, max_new_tokens=100, do_sample=False,
                return_dict_in_generate=True, output_logits=True)
    str_out = self.processor.decode(generation['sequences'][0][input_len:-1]) # [0] could be a batch index
    logits = generation['logits'][0][0] 
    return {'Q':query,'A':str_out,'scores': logits}
  
  def get_model_output(self,img_path : str, query : str ,generation_kwargs : dict):
    '''Not tested yet.
    Returns transformers ModelOutput object, and the inputs length.
    '''
    inputs= self.get_inputs(img_path,query)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generation = self.model.generate(**inputs, **generation_kwargs)
    return generation, input_len

