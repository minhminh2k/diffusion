from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from src.models.ddpm_module import DDPMModule
from src.models.ddim_sample import DDIMSample
import torch
import imageio
import numpy as np
from PIL import Image

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

# Reference: https://github.com/PAD2003/diffusion-models

def inference(cfg: DictConfig):
    DDPM_model = DDPMModule.load_from_checkpoint(cfg.ckpt_path)
    
    # sample = DDIMSample(
    #     model=DDPM_model,
    #     S=100,
    #     ddim_discretize="uniform",
    #     ddim_eta=1
    # )
    
    for S in cfg.S:
        for eta in cfg.eta:
            sample = DDIMSample(
                model=DDPM_model,
                S=S,
                ddim_discretize="uniform",
                ddim_eta=eta
            )
            
            # Generate the sample
            gen_samples = sample.generate_sample()
            gen_samples_np = gen_samples.to("cpu").numpy()

            frames = []
            for frame_idx in range(gen_samples_np.shape[0]):
                # Sequences 3x3 grid
                grid = np.vstack([np.hstack([gen_samples_np[frame_idx, i, j, :, :, 0] for j in range(3)]) for i in range(3)])
                frames.append(grid)

            # Save the GIF
            file_path = f"assets/gif/ddim_eta={eta}_S={S}.gif"
            imageio.mimsave(file_path, frames, fps=5)

@hydra.main(version_base="1.3", config_path="../configs", config_name="ddim_sampling.yaml")
def main(cfg: DictConfig):
    inference(cfg)

if __name__ == "__main__":
    torch.manual_seed(211)
    main()
