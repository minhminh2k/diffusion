from tqdm import tqdm
from src.models.components.unet.unet_basic import BasicUnet
import torch
from torch import nn
from typing import Tuple, List
import numpy as np
import wandb
import imageio
import math

'''
Diffusion
    - Denoise network : Unet
    - Alpha, Beta, Alpha bar
    - Get t 
    - Forward
    - Denoise
    - Visualize
'''

class DiffusionBasicModel(nn.Module):
    
    def __init__(
        self,
        denoise_net : BasicUnet,
        beta_start : 0.0001,
        beta_end : 0.02,
        time_steps : 1000,
        img_dims: Tuple[int, int, int] = [1, 32, 32],
        img_depth: int = 3,
        full_size: int = 32 * 32,
        dataset_name: str = "mnist",
    ) -> None : 
        
        super().__init__()
        
        self.time_steps = time_steps
        self.denoise_net = denoise_net
        self.img_dims = img_dims
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_depth = img_depth
        self.full_size = full_size
        self.dataset_name = dataset_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.beta, self.alpha, self.alpha_bar = self.create_parameter()
    
    def create_parameter(self):
        beta = torch.linspace(self.beta_start, self.beta_end,self.time_steps).to(self.device)
        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim = 0)
        return beta, alpha, alpha_bar
        
        
    def batch_dimention(self,vals, t, x_shape):
        # Get sample t 
        batch_size = t.shape[0]
        out = vals.to("cpu").gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def forward(self,x_0):
        # print(type(x_0)) # Torch.Tensor
        t = torch.randint(0,self.time_steps, (x_0.shape[0],), device = self.device).long()
        alpha_bar_t = self.batch_dimention(self.alpha_bar,t,x_0.shape)
        noise = torch.randn_like(x_0).to(self.device)
        x_t = torch.sqrt(alpha_bar_t).to("cpu")*x_0.to("cpu") + torch.sqrt(1 - alpha_bar_t).to("cpu")*noise.to("cpu")
        noise = noise.to(self.device)
        x_t = x_t.to(self.device)

        self.denoise_net.to(self.device)
        noise_pred = self.denoise_net(x_t,t)
        return noise_pred, noise
    
    @torch.no_grad()
    def sample(self, x, t ):
        with torch.no_grad():
            alpha_t = self.batch_dimention(self.alpha, t, x.shape)
            alpha_bar_t = self.batch_dimention(self.alpha_bar, t, x.shape)
            beta_t = self.batch_dimention(self.beta, t, x.shape)
            z = torch.randn(x.shape).to(self.device)
            e_pred = self.denoise_net(x,t).to(self.device)
            pre_scale = (1/torch.sqrt(alpha_t)).to(self.device)
            e_scale = ((1- alpha_t)/torch.sqrt(1 - alpha_bar_t)).to(self.device)
            sigma = torch.sqrt(beta_t).to(self.device)
            x = pre_scale*(x - e_scale*e_pred) + sigma*z
            return x
            
    def sequence_sample(self):
        gif_shape = [1, 9]
        sample_batch_size = gif_shape[0] * gif_shape[1]
        n_hold_final = 10
        gen_samples = []
        x = torch.randn((sample_batch_size, self.img_depth,int(math.sqrt(self.full_size)), int(math.sqrt(self.full_size)))).to(self.device)
        
        # sample_steps = tqdm(
        #             (torch.linspace(0, 1000 - 1,1000).flip(0).to(torch.int64)),
        #             desc="Sampling t")
        sample_steps = torch.linspace(0, 1000 - 1,1000).flip(0).to(torch.int64)

        for i, t in enumerate(sample_steps):
          t_ = t
          t = torch.full((x.shape[0], ),
                                   t,
                                   device=self.device,
                                   dtype=torch.int64)

          x = self.sample(x,t)
          if t_  % 10 == 0 :
            gen_samples.append(x)
            
        gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)
        gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2

        gen_samples = (gen_samples * 255).type(torch.uint8)
        gen_samples = gen_samples.reshape(-1, gif_shape[0], gif_shape[1], int(math.sqrt(self.full_size)), int(math.sqrt(self.full_size)), self.img_depth)
        
        # Shape
        print(gen_samples.shape) # 100, 1, 9, 32, 32, 1
        
        gen_samples_np = gen_samples.to("cpu").numpy()
        frames = []

        for frame_idx in range(gen_samples_np.shape[0]):
            # Arrange sequences in a 3x3 grid for each frame
            grid = np.vstack([np.hstack([gen_samples_np[frame_idx, i, j, :, :, 0] for j in range(9)]) for i in range(1)])
            frames.append(grid)
        return frames
    
    def visualize(self):
        frames = self.sequence_sample()
        file_path =f"assets/{self.dataset_name}.gif"
        
        imageio.mimsave(file_path, frames, fps=10)
        # wandb.log({"samples": wandb.Image(file_path)})
        return file_path