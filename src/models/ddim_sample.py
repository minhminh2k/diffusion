from typing import Any, Dict, List, Tuple

import hydra
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

import imageio
import numpy as np
from PIL import Image

from src.models.ddpm_module import DDPMModule
import torch
import math

# From PAD

class DDIMSample:
    def __init__(
        self,
        model: DDPMModule,
        S: int, 
        ddim_discretize: str = "uniform", 
        ddim_eta: float = 0,
    ):
        self.S = S
        self.model = model
        self.ddim_eta = ddim_eta
        
        # Timesteps
        t_range = model.t_range
        if ddim_discretize == "uniform":
            c = model.t_range // S
            self.time_steps = np.asarray(list(range(0, t_range, c))) + 1
        elif ddim_discretize == "quad":
            self.time_steps = ((np.linspace(0, np.sqrt(t_range * .8), S)) ** 2).astype(int) + 1
        else:
            raise NotImplementedError(ddim_discretize)
            
    
    def sigma(self, t, t_prev):
        alpha_bar_t = self.model.alpha_bar(t)
        alpha_bar_t_prev = self.model.alpha_bar(t_prev)
        sigma = (self.ddim_eta * ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)) ** .5)
        return sigma
        
    def denoise_sample(self, x_t, t, t_prev):
        
        # Elements
        epsilon_theta = self.model.forward(x_t, torch.full((x_t.shape[0], 1), t, device=self.model.device))
        alpha_bar_t = self.model.alpha_bar(t)
        alpha_bar_t_prev = self.model.alpha_bar(t_prev)
        sigma_t = self.sigma(t, t_prev)
        
        # Components
        pred_x0 = (x_t - math.sqrt(1 - alpha_bar_t) * epsilon_theta) / math.sqrt(alpha_bar_t)
        if self.ddim_eta == 0:
            pred_x0 = pred_x0.clamp(-1, 1)
            
        dir_xt = math.sqrt(1 - alpha_bar_t_prev - sigma_t ** 2) * epsilon_theta
        
        if sigma_t == 0. or t_prev == 0: # Tensor
            noise = 0.
        else:
            noise = torch.randn(x_t.shape, device=x_t.device)
        
        # Caculate
        x_t_prev = math.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + sigma_t * noise
        
        return x_t_prev
    
    def generate_sample(self):
        print(len(self.time_steps))
        
        # config
        gif_shape = [3, 3]
        sample_batch_size = gif_shape[0] * gif_shape[1]
        n_hold_final = 10
        
        # generation process
        gen_samples = []
        x = torch.randn((sample_batch_size, self.model.img_depth, int(math.sqrt(self.model.in_size)), int(math.sqrt(self.model.in_size))), device=self.model.device)
        
        for i, t, t_prev in zip(range(len(self.time_steps)), np.flip(self.time_steps[1:]), np.flip(self.time_steps[:-1])):
            x = self.denoise_sample(x, t, t_prev)
            if i % (len(self.time_steps) / 20) == 0:
                gen_samples.append(x)
        for _ in range(n_hold_final):
            gen_samples.append(x)

        # post - process
        gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1) # (frame, 9, width, height)
        gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2 # (frame, 9, width, height)
        gen_samples = (gen_samples * 255).type(torch.uint8)
        gen_samples = gen_samples.reshape(-1, gif_shape[0], gif_shape[1], int(math.sqrt(self.model.in_size)), int(math.sqrt(self.model.in_size)), self.model.img_depth) # (frame, 3, 3, width, height, depth)
        
        return gen_samples