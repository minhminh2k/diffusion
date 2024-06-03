import torch
from lightning import LightningModule
from torchvision import models
from torch import nn

class ClassifierCNN(nn.Module):
    """A simple fully-connected neural net for computing predictions."""
    
    def __init__(self, input_size: int = 28,
                 num_classes: int = 10,
                 pretrained_model: str = 'resnet34',
    ) -> None:
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes

        if pretrained_model is not None:
            self.pretrained_model = pretrained_model
        else:
            self.name_pretrained = 'resnet34'

        self.net = models.get_model(name=self.pretrained_model, weights="DEFAULT")
        
        self.last_layer = nn.Linear(1000, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return self.last_layer(x)
    
if __name__ == "__main__":
    _ = ClassifierCNN()