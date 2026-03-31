import torch
from models.model import Model
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from models.internvl_conversation import get_conv_template

class InternVLModel(Model):

  def __init__(self,model_id = "OpenGVLab/InternVL2_5-1B",**kwargs) -> None:
    self.__dict__.update(kwargs)
    self.model_id = model_id
    self.model, self.processor, self.vocab = self._get_model(model_id)

  def _get_model(self,model_id):
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

    model = AutoModel.from_pretrained(
        model_id, device_map="auto",
        attn_implementation = 'eager', # needed to use output_attention
        trust_remote_code = True
    ).eval()
    processor = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    #tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    vocab = processor.get_vocab()
    return model,processor, vocab

  ###############
  ## Inference ##
  ###############

  def get_inputs(self,img_path,query_string):
    '''Applies the processor and adds system message to the query.
    '''
    #def get_inputs(self,tokenizer,pixel_values,question,IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
    #         verbose=False):
    tokenizer = self.processor
    pixel_values = load_image(img_path,input_size=448)

    IMG_START_TOKEN='<img>'
    IMG_END_TOKEN='</img>'
    IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
    
    if pixel_values is not None and '<image>' not in query_string:
        question = '<image>\n' + query_string
    else:
        question = query_string
    if True:#num_patches_list is None:
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    self.model.img_context_token_id = img_context_token_id

    template = get_conv_template(self.model.template)
    template.system_message = self.model.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors='pt')
    model_inputs['input_ids']=model_inputs['input_ids'].to(self.model.device)
    model_inputs['pixel_values']=pixel_values.to(self.model.device)
    return model_inputs

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


###################################
## IMAGE PREPROCESSING FUNCTIONS ##
#internVL splits big images in blocks of 448×448 pixels;
#the inputs are also normalized

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    if type(image_file)==str:
       image = Image.open(image_file).convert('RGB')
    else:
       image = image_file
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values