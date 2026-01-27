import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw
import hydra
from omegaconf import OmegaConf, DictConfig
import os
from tqdm import tqdm
from ast import literal_eval

import pyrootutils, sys
pyrootutils.setup_root('.', dotenv=True, pythonpath=False)
sys.path.append('./src')

@hydra.main(version_base=None, config_path='config', config_name='dataset_gen')
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
  


  datadir = f"{cfg.dataset["DATAPATH"]}{cfg.dataset["DATASET_NAME"]}/"
  imagedir = datadir+"img/"
  configpath = datadir+"config.yaml"

  #############################
  ## CREATE DATA DIRECTORIES ##
  if not os.path.exists(datadir):
    os.mkdir(datadir)
    os.mkdir(imagedir)
    print("Data directory created")
  else:
    if not os.path.exists(imagedir):
      os.mkdir(imagedir)
      print("Data directory already existing, img directory created")
    else:
      print("Data and image directory already existent, overwriting images")

  ##############################
  ## GENERATE/IMPORT METADATA ##
  if not cfg.recreate_meta and os.path.exists(datadir+'metadata.csv'): # don't recreate metadata if already existent
    print("Importing pre-made metadata")
    metadata = pd.read_csv(datadir+'metadata.csv', dtype={'ID': 'string'}, 
                    converters={'target_pos': literal_eval, 
                                'distractors': literal_eval}
                    ).to_dict('records')
  else:
    print('Creating metadata')
    meta_generator = hydra.utils.instantiate(cfg.dataset.generate_metadata_function)
    metadata = meta_generator(cfg.dataset,constants)
   pd.DataFrame(metadata).to_csv(datadir+"metadata.csv",index=False) #save metadata
    with open(configpath,'w') as f: #save config used to generate it
      OmegaConf.save(cfg,f)#, default_flow_style=False)

  if not cfg.generate_images:
    print(len(metadata),"images will NOT be created.")
    return

  print(len(metadata),"images will be created")

  #################################
  ## CREATE IMAGES AND SAVE DATA ##
  image_maker = hydra.utils.instantiate(cfg.dataset.make_image_function)
  for meta in tqdm(metadata,desc='Creating images'): # create and save images .png
    img = image_maker(meta,cfg.dataset,constants)
    img.save(f"{imagedir}/{meta["ID"]}.png")

if __name__== "__main__":
  main()
