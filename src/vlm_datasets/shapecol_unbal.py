import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw
from tqdm import tqdm

from itertools import product
import random

# import pyrootutils, sys
# pyrootutils.setup_root(search_from='.', dotenv=True, pythonpath=False)
# sys.path.append('./lib')



def n_polygon_array(n : int) -> np.array:
  'Returns the array version of an image of the specified polygon, with 1-0 values shape-background'
  a = Image.new('RGB', (121,121), 'white')
  b = ImageDraw.Draw(a)
  b.regular_polygon((60,60,60),n,fill=mcolors.to_hex('black'))
  shape =(np.array(a))
  return shape

def generate_targets(config) -> list:
    metadata = []
    for t_col in OmegaConf.to_container(config["COLORS_DICT"]):
        for t_shape in config["SHAPES"]:
            for numerosity in config["N_distractors"]:
                for pia in config["P_interfere"]:
                  if (numerosity==2) and pia in [0.25,0.75]:## 2 distractors don't allow these P_int values
                     continue
                  for nuc in config["N_unique_distractors"]:
                    for has_target in [True, False]:
                      for pref in ['col','sha']:
                        for run in range(config["REPEATS"]):
                            row = {
                                "ID": str(len(metadata)).zfill(4),  # Incremental number padded to 4 digits
                                "target_color": t_col,
                                "target_shape": t_shape,
                                "has_target": has_target, 
                                #"target_pos": [None,None], will be many, stored in distractors
                                "N_distractors": numerosity, 
                                "P_interfere": pia,
                                "N_unique_distractors": min(nuc,numerosity),
                                "number": run,
                                "pref": pref,
                                "distractors": [],
                            }
                            metadata.append(row)
    return  metadata

def generate_positions(n_objects, canvas_size, stencil_size) -> np.ndarray:
    '''Generates random non-overlapping positions for n_objects of size stencil_size×stencil_size.

    Returns
    -------
    np.ndarray
      Array of positions [x,y].
    '''
    # Calculate valid position range
    margin = stencil_size // 2
    min_x = margin
    max_x = canvas_size[0] - margin
    min_y = margin
    max_y = canvas_size[1] - margin
    
    # Pre-allocate positions array
    positions = np.zeros((n_objects, 2),dtype=int)
    
    # Generate valid positions
    for i in range(n_objects):
        while True:
            pos = np.random.randint([min_x, min_y], [max_x, max_y], size=2)
            if i == 0:  # First object can go anywhere
                positions[i] = pos
                break
                
            # Check distance from all previous objects
            #distances = np.linalg.norm(positions[:i] - pos, axis=1)

            distances = (np.abs(positions[:i] - pos))#[prev][x/y]
            dist_is_ok = np.any(distances >= stencil_size,axis=1)
            if np.all(dist_is_ok):#np.all(distances >= stencil_size+10): #5 buffer
                positions[i] = pos
                break
    
    # Shuffle positions before returning
    np.random.shuffle(positions)
    return positions#,#np.linalg.norm(positions - positions[0], axis=1)

def generate_objects(meta : dict, config: dict) -> None:
  '''Generates color and shape for all objects, saving them in the metadata.

  Note: deletes all previously stored distractors.'''
  ## how many of each distractor object type
  n_identical_objects = (meta['N_distractors'])/(meta['N_unique_distractors']) #one is target-replacer, treated separately
  assert (n_identical_objects%1) == 0.
  n_identical_objects = int(n_identical_objects)
  # how many types of conj distractors, and disj ones
  n_unique_conj_distractors = meta['P_interfere']*meta['N_unique_distractors']
  assert n_unique_conj_distractors%1 == 0.
  n_unique_conj_distractors = int(n_unique_conj_distractors)
  n_unique_disj_distractors = meta['N_unique_distractors']-n_unique_conj_distractors

  n_distractor_colors = 2 ## how many colors/shapes to use that are not the target color
  
  used_colors = set() #keep track of how many colors were used
  used_shapes = set()

  ## select the N_unique_objects colorshapes to use
  meta['distractors'] = []
  # colors/shapes distinct from target
  dis_shapes = (config["SHAPES"]).copy()
  dis_colors = (config["COLORS"]).copy()
  dis_shapes.remove(meta['target_shape'])
  dis_colors.remove(meta['target_color'])

  dis_shapes = random.sample(dis_shapes,n_distractor_colors)
  dis_colors = random.sample(dis_colors,n_distractor_colors)

  
  ### TARGET
  if meta['has_target']:
    # used_colors.add(meta['target_color'])
    # used_shapes.add(meta['target_shape'])
    d = {
      'color': meta['target_color'], ##could be chosen before to have only one distractor color
      'shape': meta['target_shape'],
    }
    meta['distractors'].append(d)
  

  
  ### CONJUNCTIVE DISTRACTORS
  conj_color = [col_shape for col_shape in product([meta['target_color']],dis_shapes)]
  random.shuffle(conj_color)
  conj_shape = [col_shape for col_shape in product(dis_colors,[meta['target_shape']])]
  random.shuffle(conj_shape)
  conj_color_shapes = [None]*(len(conj_color)+len(conj_shape))
  if meta['pref'] == 'col': #first in line will be same color, different shape
    conj_color_shapes[::2] = conj_color
    conj_color_shapes[1::2]= conj_shape
  else:
    conj_color_shapes[1::2]= conj_color
    conj_color_shapes[::2] = conj_shape
  # Add conjunctive distractors following their order
  for i in range(n_unique_conj_distractors):
    conj_color_shape = conj_color_shapes[i]
    if conj_color_shape[0]!=meta['target_color']: used_colors.add(conj_color_shape[0])
    if conj_color_shape[1]!=meta['target_shape']: used_shapes.add(conj_color_shape[1])
    for _ in range(n_identical_objects):
      d = {
        'color': conj_color_shape[0], ##could be chosen before to have only one distractor color
        'shape': conj_color_shape[1],
      }
      meta['distractors'].append(d)
  
  ### DISJUNCTIVE DISTRACTORS
  disj_color_shapes = [col_shape for col_shape in product(dis_colors,dis_shapes)] #lists of tuples (color,shape)
  for i in range(n_unique_disj_distractors):
    if disj_color_shapes: #only if it's not empty
      if (len(used_colors)==n_distractor_colors and len(used_shapes)==n_distractor_colors) or meta['P_interfere']==0.:
        ## if 3 colors and shapes were already used, we can choose randomly
        dist_color_shape = random.choice(disj_color_shapes)
      elif len(used_colors)<n_distractor_colors and len(used_shapes)< n_distractor_colors:
        ## se ho un colore-forma ancora da usare, scegli tra quelli non ancora usati
        unused_colors = list( set(dis_colors.copy()).difference(used_colors) )
        unused_shapes = list( set(dis_shapes.copy()).difference(used_shapes) )
        ## when P_interfere is low, only 2 colors/shapes may be available; hence we
        ## must add one more
        if not unused_colors or not unused_shapes:
           # we get here with P=0 (only 2 disj colors possible)
           dist_color_shape = random.choice(disj_color_shapes)
        ## combinazioni di questi sicuramente non sono ancora state usate
        else:
          dist_color_shape = (random.choice(unused_colors),
                            random.choice(unused_shapes))
      elif len(used_colors)<n_distractor_colors and len(used_shapes)==n_distractor_colors:
        unused_colors = list(set(dis_colors).difference(used_colors))

        dist_color_shape = (random.choice(unused_colors),
                            random.choice(dis_shapes))
      elif len(used_colors)==n_distractor_colors and len(used_shapes)<n_distractor_colors:
        unused_shapes = list(set(dis_shapes).difference(used_shapes))
        dist_color_shape = (random.choice(dis_colors),
                            random.choice(unused_shapes))

      used_colors.add(dist_color_shape[0])
      used_shapes.add(dist_color_shape[1])
      try:
        disj_color_shapes.remove(dist_color_shape)
      except:
        raise Exception
    else:# if empty, we have to get another color-shape to keep the number at three
      # this should happen only if P_interfere_absolute = 0
      # this can be avoided if n_unique_shapes is less then (unique_colors -1)*(unique_shapes-1)
      #get the remaining colors and shapes
      # cols = (config['COLORS']).copy()
      # cols.remove(meta['target_color'])
      # for col in dis_colors:
      #   cols.remove(col)
      # shas = config['SHAPES'].copy()
      # shas.remove(meta['target_shape'])
      # for shape in dis_shapes:
      #   shas.remove(shape)
      # if meta['P_interfere_absolute']== 0. :
      #   dist_color_shape = (random.choice(cols),random.choice(shas))
      # else:
      print(meta['distractors'])
      raise Exception('P_interfere not zero, but all objects were used?')
    
    for n in range(n_identical_objects):
        d = {
          'color': dist_color_shape[0], ##could be chosen before to have only one distractor color
          'shape': dist_color_shape[1],
        }
        meta['distractors'].append(d)

def color_shape(img: np.ndarray,stencil_color: str,background_color: str='white') -> np.ndarray:
    ''' Colors a grayscale numpy array. '''
    rgbcolor = lambda x : mcolors.to_rgb(x) if type(x)== str else x
    fg_col = np.array(mcolors.to_rgb(stencil_color)).astype(np.float32)
    bg_col = np.array(mcolors.to_rgb(background_color)).astype(np.float32)
    img = img.astype(np.float32) / 255  # Normalize grayscale
    colored = bg_col.reshape(3, 1, 1) + (fg_col.reshape(3, 1, 1) - bg_col.reshape(3, 1, 1)) * (1 - img)
    return (255 * colored).astype(np.uint8)

def rgb256_color_shape(img: np.ndarray,stencil_color : list , background_color: list=[255,255,255]):
  '''Colors a 0-1 shape with stencil_color, in rgb format (eg red may be [255,0,0])'''
  fg_col = (np.array(stencil_color)/255.).astype(np.float32)
  bg_col = (np.array(background_color)/255.).astype(np.float32)
  img = img.astype(np.float32) / 255  # Normalize grayscale
  colored = bg_col.reshape(3, 1, 1) + (fg_col.reshape(3, 1, 1) - bg_col.reshape(3, 1, 1)) * (1 - img)
  return (255 * colored).astype(np.uint8)


### The following are the only 2 functions called outside, selected in config

def make_image_from_metadata(meta: dict, config : dict, constants : dict) -> Image.Image:
  canvas = Image.new('RGB', constants['CANVAS_SIZE'], 'white')

  for i in range(0, len(meta['distractors'])):
    shape_img = constants['SHAPES_NP'][meta['distractors'][i]['shape']]
    #rgb_color = np.array([int(255 * x) for x in mcolors.to_rgb(meta['target_color'])])
    colored_shape = rgb256_color_shape(shape_img.transpose(), config['COLORS_DICT'][meta['distractors'][i]['color']])#rgb_color)  # Returns (3, H, W)
    shape_pil = Image.fromarray(colored_shape.transpose())
    if shape_pil.size != (constants['STENCIL_SIZE'], constants['STENCIL_SIZE']):
        shape_pil = shape_pil.resize((constants['STENCIL_SIZE'], constants['STENCIL_SIZE']))
    paste_x = meta['distractors'][i]['pos'][0] - constants['STENCIL_SIZE'] // 2
    paste_y = meta['distractors'][i]['pos'][1] - constants['STENCIL_SIZE'] // 2
    canvas.paste(shape_pil, (paste_x, paste_y))
  return canvas

def generate_metadata(meta_config, constants) -> list:
  '''Generate metadata dictionaries given the configurations.
  
  Returns:
  --------
  metadata : list of dict
    A list of metadata dictionaries, one for each data entry.'''
  metadata = generate_targets(meta_config)
  for meta in tqdm(metadata, desc='Adding distractors'):
    generate_objects(meta,meta_config)
    #print(meta['N_objects'])
    pos = generate_positions(len(meta['distractors']),canvas_size=constants['CANVAS_SIZE'],stencil_size=constants['STENCIL_SIZE'])
    for n,d in enumerate(meta['distractors']):
      d['pos']=pos[n].tolist()
  return metadata