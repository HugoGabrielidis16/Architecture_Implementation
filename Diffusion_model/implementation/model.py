import torch
import torch.nn as nn
from embedding import PositionalEmbedding 




"""
The model described in the 'Denoising Diffusion Probabilistic Models' paper is a 'U-NET backbone based on wide ResNet',
they remplaced the weight normalization layer with group normalization

32x32 mode has four features maps.
All models have two convolutional residual blocks per resolution level and self-attention blocks at the 16 × 16 resolution between the
convolutional blocks 
"""
class Diffusion_model(nn.Module):
    def __init__(self):
        super(Diffusion_model, self).__init__()