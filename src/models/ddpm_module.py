from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
import math
import torch.nn as nn
import numpy as np
import wandb
import imageio
import torchvision.utils as vutils
from torchvision.transforms import ToPILImage

# Reference: https://github.com/awjuliani/pytorch-diffusion

class DDPMModule(LightningModule):
    def __init__(
        self,
        beta_small: float,
        beta_large: float,
        in_size: int, # width * height
        t_range: int,
        img_depth: int, # channel
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        dataset_name: str,
    ) -> None:
        super().__init__()
        
        self.beta_small = beta_small
        self.beta_large = beta_large
        self.t_range = t_range
        self.in_size = in_size
        self.img_depth = img_depth
        self.dataset_name = dataset_name
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        
        self.save_hyperparameters(logger=False)

        self.net = net(img_depth=self.img_depth)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x: torch.Tensor, t: int) -> torch.Tensor:
        return self.net(x, t)

    def on_train_start(self) -> None:
        self.val_loss.reset()
    
    
    ##############################################################################################################################################
    ############################################################### DIFFUSION STEP ###############################################################
    
    def beta(self, t):
        return self.beta_small + (t / self.t_range) * (self.beta_large - self.beta_small)

    def alpha(self, t):
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        return math.prod([self.alpha(j) for j in range(t)])

    def model_step(self, batch, batch_idx): 
        """
        Corresponds to Algorithm 1 from (Ho et al., 2020).
        """
        x = batch[0]
        # print(batch_size.shape) # tensor 128, 1, 32, 32
        # print(type(batch[0])) # Torch.Tensor
        
        batch_size = x.shape[0]

        ts = torch.randint(low=0, high=self.t_range, size=(batch_size, ), device=self.device)
        
        # ts = torch.randint(low=0, high=self.t_range, size=(batch_size,)).to(self.device)
        
        noise_imgs = []
        epsilons = torch.randn(size=x.shape).to(self.device)
        
        for i in range(batch_size):
            alpha_bar_t = self.alpha_bar(ts[i])
            x_0 = x[i]
            x_t = math.sqrt(alpha_bar_t) * x_0 + math.sqrt(1 - alpha_bar_t) * epsilons[i]
            noise_imgs.append(x_t)
        
        noise_imgs = torch.stack(tensors=noise_imgs, dim=0)

        # Reverse the diffusion process 
        epsilon_thetas = self.forward(noise_imgs, ts.unsqueeze(-1).type(torch.float))

        # Calculate the MSE loss
        loss = nn.functional.mse_loss(
            input=epsilon_thetas.reshape(-1, self.in_size),
            target=epsilons.reshape(-1, self.in_size)
        )

        return loss

    def denoise_sample(self, x_t, t):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        This function is denoising process from x_{t} to x_{t-1}
        """
        with torch.no_grad():
            if t > 1:
                z = torch.randn(size=x_t.shape, device=self.device)
            else:
                z = 0

            # Caculate x_(t-1)
            sigma_t = math.sqrt((1.0 - self.alpha_bar(t-1)) / (1.0 - self.alpha_bar(t)) * self.beta(t))
            epsilon_theta = self.forward(x_t, t.unsqueeze(-1).type(torch.float))
            A = 1 / math.sqrt(self.alpha(t))
            B = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
            C = sigma_t * z

            x = A * (x_t - B * epsilon_theta) + C # x (t-1)
            return x
    
    def generate_sample(self):
        # Configuration
        gif_shape = [3, 3] # 3, 3
        sample_batch_size = gif_shape[0] * gif_shape[1]
        n_hold_final = 10
        
        # Generate samples from denoising process 
        gen_samples = []
        x = torch.randn((sample_batch_size, self.img_depth, int(math.sqrt(self.in_size)), int(math.sqrt(self.in_size))), device=self.device)
        sample_steps = torch.arange(self.t_range - 1, 0, -1, device=self.device)
        for t in sample_steps:
            x = self.denoise_sample(x, t)
            if t % 50 == 0:
                gen_samples.append(x)
        for _ in range(n_hold_final):
            gen_samples.append(x)
        
        # post - process
        gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1) # (frame, 9, width, height)
        gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2 # (frame, 9, width, height)
        gen_samples = (gen_samples * 255).type(torch.uint8)
        gen_samples = gen_samples.reshape(-1, gif_shape[0], gif_shape[1], int(math.sqrt(self.in_size)), int(math.sqrt(self.in_size)), self.img_depth) # (frame, 3, 3, width, height, depth)
        
        print(gen_samples.shape) # torch.Size([29, 3, 3, 32, 32, 1])
        
        return gen_samples
            
    ############################################################### DIFFUSION STEP ###############################################################
    ##############################################################################################################################################
    
    def on_train_start(self) -> None:
        # Fixed Noise for visualize
        n_samples = 25

        # log sample real images for visualization
        train_loader = self.trainer.train_dataloader

        # EuroSAT
        x, y = next(iter(train_loader))
        
        samples = x[:n_samples]
        grid = vutils.make_grid(samples, nrow=int(np.sqrt(n_samples)), normalize=True)
        self.logger.log_image(key='Real images', images=[ToPILImage()(grid)])


    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss= self.model_step(batch, batch_idx)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss= self.model_step(batch, batch_idx)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self) -> None:
        gen_samples = self.generate_sample()
        gen_samples_np = gen_samples.to("cpu").numpy()

        # Stack
        frames = []
        if self.img_depth == 1:
            for frame_idx in range(gen_samples_np.shape[0]):
                # Arrange sequences in a 3x3 grid for each frame
                grid = np.vstack([np.hstack([gen_samples_np[frame_idx, i, j, :, :, 0] for j in range(3)]) for i in range(3)]) # (32 x 32)
                frames.append(grid)
        if self.img_depth == 3:
            for frame_idx in range(gen_samples_np.shape[0]):
                # Arrange sequences in a 3x3 grid for each frame
                grid = np.vstack([np.hstack([gen_samples_np[frame_idx, i, j, :, :, :] for j in range(3)]) for i in range(3)]) # (32 x 32)
                frames.append(grid)
        
        # Save the GIF
        file_path = f"assets/{self.dataset_name}.gif"
        imageio.mimsave(file_path, frames, fps=5)
        
        wandb.log({"samples": wandb.Image(file_path)})

        
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == "__main__":
    _ = DDPMModule(None, None, None, None)