from torch import nn, optim
import torch
from torch.nn import functional as F
from typing import Any, Callable, Optional
import math

class VanillaGAN(nn.Module):
    def __init__(self, resolution, latent_dim, hidden_dim=512, channels=3):
        super(VanillaGAN, self).__init__()
        output_dim = resolution * resolution * channels

        self.layers = nn.Sequential(
            self.gen_block(latent_dim, hidden_dim),
            self.gen_block(hidden_dim, hidden_dim*2),
             self.gen_block(hidden_dim*2, hidden_dim*2),
            self.gen_block(hidden_dim*2, hidden_dim),
            self.gen_block(hidden_dim, hidden_dim),
            self.gen_block(hidden_dim, hidden_dim//2),
            
            nn.Linear(hidden_dim//2, output_dim),
            nn.Tanh()
        )

    def gen_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim, 0.8),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.layers(x)