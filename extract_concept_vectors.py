"""extract_concept_vectors.py

Reads trained probe .nn files, extracts normalized concept vectors, and saves
them as .pt files ready for analysis notebooks and the steering pipeline.

Replaces the manual cells in 4_2_a_probe.ipynb with a reproducible script.

Outputs
-------
For probe type ``independent`` (probe=attn_single):
  outputs/<model_name>/colorshape_cv_probe.pt
      Raw L2-normalized probe concept vectors  (36, D).
  outputs/<model_name>/colorshape_cv_probepca.pt
      PCA-regularized vectors projected onto the 2N-2 principal components
      of the probe vector set, then re-normalized  (36, D).
  outputs/temp_eval/<model_name>/template_newprobe.pt     (copy of cv_probe)
  outputs/temp_eval/<model_name>/template_probepca.pt     (copy of cv_probepca)

For probe type ``joint`` (probe=attn_joint):
  outputs/<model_name>/colorshape_cv_jointprobe.pt
      L2-normalized rows of the joint probe's attn_proj.weight  (36, D).
  outputs/temp_eval/<model_name>/template_jointprobe.pt   (copy of cv_jointprobe)

Usage
-----
  python -m extract_concept_vectors +experiment=bal_qwen7b probe=attn_single +layer=mmp
  python -m extract_concept_vectors +experiment=bal_qwen7b probe=attn_joint  +layer=mmp

  # With external output root:
  python -m extract_concept_vectors +experiment=bal_qwen7b probe=attn_single +layer=mmp \\
      output_root=outputs/
"""

import os
import shutil
import glob

import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
from omegaconf.errors import ConfigAttributeError
from itertools import product

import pyrootutils, sys
import warnings
warnings.filterwarnings('ignore')

pyrootutils.setup_root('.', dotenv=True, pythonpath=False)
sys.path.append('./src')


# Number of PCA components for the PCA-regularized variant.
# 2N - 2 for N categories (6 colors, 6 shapes → 10).
N_PCA_COMPONENTS = 10


def _save(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(tensor, path)
    print(f"  Saved {path}  {tuple(tensor.shape)}")


def _copy_as_template(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)
    print(f"  Template → {dst}")


def extract_independent(cfg, probeoutputdir, cv_outdir, template_outdir):
    """Load 36 individual .nn files and produce raw + PCA concept vectors."""
    colors = cfg.dataset.COLORS.copy()
    shapes = cfg.dataset.SHAPES.copy()
    versionlabel = cfg.probe.versionlabel

    raw_vecs = []
    for color in colors:
        for shape in shapes:
            nn_path = os.path.join(probeoutputdir, f'{color}{shape}_{versionlabel}.nn')
            if not os.path.exists(nn_path):
                raise FileNotFoundError(
                    f"Probe file not found: {nn_path}\n"
                    "Run train_probes.py first, or check output_root and layer."
                )
            sd = torch.load(nn_path, map_location='cpu')
            # attn_proj.weight shape: (1, D) — squeeze to (D,)
            raw_vecs.append(sd['attn_proj.weight'].detach().squeeze().double())

    raw_vecs = torch.stack(raw_vecs)  # (36, D)

    # --- Raw probe vectors ---
    concept_vectors = F.normalize(raw_vecs, dim=1).float()
    cv_probe_path = os.path.join(cv_outdir, 'colorshape_cv_probe.pt')
    _save(concept_vectors, cv_probe_path)
    _copy_as_template(cv_probe_path,
                      os.path.join(template_outdir, 'template_newprobe.pt'))

    # --- PCA-regularized vectors ---
    # Project onto the top N_PCA_COMPONENTS principal directions of the probe
    # vector set, then re-normalize.  This forces the vectors to lie in the
    # structured subspace spanned by the main axes of variation, acting as a
    # geometric regularizer that suppresses discriminative shortcuts.
    _, _, v = torch.pca_lowrank(raw_vecs, q=36, center=True)
    cv_pca = F.normalize(
        concept_vectors.double() @ v[:, :N_PCA_COMPONENTS] @ v[:, :N_PCA_COMPONENTS].T,
        dim=1,
    ).float()
    cv_probepca_path = os.path.join(cv_outdir, 'colorshape_cv_probepca.pt')
    _save(cv_pca, cv_probepca_path)
    _copy_as_template(cv_probepca_path,
                      os.path.join(template_outdir, 'template_probepca.pt'))


def extract_joint(cfg, probeoutputdir, cv_outdir, template_outdir):
    """Load the single joint .nn file and produce normalized concept vectors."""
    versionlabel = cfg.probe.versionlabel
    nn_path = os.path.join(probeoutputdir, f'joint_{versionlabel}.nn')
    if not os.path.exists(nn_path):
        raise FileNotFoundError(
            f"Joint probe file not found: {nn_path}\n"
            "Run train_probes.py with probe=attn_joint first."
        )

    sd = torch.load(nn_path, map_location='cpu')
    # attn_proj.weight shape: (K, D) = (36, D) — rows are per-concept vectors
    raw_vecs = sd['attn_proj.weight'].detach().double()  # (36, D)

    concept_vectors = F.normalize(raw_vecs, dim=1).float()
    cv_joint_path = os.path.join(cv_outdir, 'colorshape_cv_jointprobe.pt')
    _save(concept_vectors, cv_joint_path)
    _copy_as_template(cv_joint_path,
                      os.path.join(template_outdir, 'template_jointprobe.pt'))


@hydra.main(version_base=None, config_path='config', config_name='probe_training')
def main(cfg: DictConfig) -> None:
    try:
        layer = cfg.layer
    except ConfigAttributeError:
        layer = input('Insert a layer label: ')

    # Probe directory: mirrors train_probes.py which uses cfg.probe.outputs_path
    # i.e. outputs/<model_name>/<task_type>/<dataset_name>/<layer>_probes/
    probeoutputdir = os.path.join(cfg.probe.outputs_path, f'{layer}_probes')

    cv_outdir       = os.path.join('outputs', cfg.model.model_name)
    template_outdir = os.path.join('outputs', 'temp_eval', cfg.model.model_name)

    print(f"\nModel:    {cfg.model.model_name}")
    print(f"Dataset:  {cfg.dataset.DATASET_NAME}")
    print(f"Layer:    {layer}")
    print(f"Probes:   {probeoutputdir}")
    print(f"CV out:   {cv_outdir}")
    print(f"Template: {template_outdir}\n")

    probe_type = cfg.probe.type

    if probe_type == 'independent':
        print("Extracting independent probe concept vectors ...")
        extract_independent(cfg, probeoutputdir, cv_outdir, template_outdir)

    elif probe_type == 'joint':
        print("Extracting joint probe concept vectors ...")
        extract_joint(cfg, probeoutputdir, cv_outdir, template_outdir)

    else:
        raise NotImplementedError(f"Probe type '{probe_type}' not supported.")

    print("\nDone.")


if __name__ == '__main__':
    main()
