import torch
import torch.nn as nn

from diffab.modules.common.geometry import construct_3d_basis
from diffab.modules.common.so3 import rotation_to_so3vec
from encoder.embedding import *
from diffab.utils.protein.constants import max_num_heavyatoms, BBHeavyAtom
import torch.nn.functional as F

class ContrastiveDiffAb(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heavy_encoder = diffabencoder(cfg)
        self.light_encoder = diffabencoder(cfg)
        self.antigen_encoder = diffabencoder(cfg)
        self.temperature = 0.1  # You can tune this value
    
    def forward(self, batch):
        # Get batch size and device
        batch_size = batch['heavy']['aa'].size(0)
        device = batch['heavy']['aa'].device
        
        # Encode heavy, light chains, and antigen
        heavy_feat = self.heavy_encoder(batch['heavy'])  # Shape: (N, L)
        light_feat = self.light_encoder(batch['light'])  # Shape: (N, L)
        antigen_feat = self.antigen_encoder(batch['antigen'])  # Shape: (N, L)
        
        # Transpose antigen features for batch matrix multiplication
        transposed_antigen = torch.transpose(antigen_feat, 0, 1)  # Shape: (L, N)
        
        # Compute dot products
        ha_dot = torch.matmul(heavy_feat, transposed_antigen)  # Shape: (N, N)
        hl_dot = torch.matmul(light_feat, transposed_antigen)  # Shape: (N, N)
        
        # Apply softmax to each row to get probabilities
        ha_prob = F.softmax(ha_dot, dim=1)  # Shape: (N, N)
        hl_prob = F.softmax(hl_dot, dim=1)  # Shape: (N, N)
        
        # Create target labels which should be the indices of diagonal elements
        target = torch.arange(batch_size, device=device)  # Shape: (N,)
        
        # Compute negative log likelihood loss for diagonal elements
        ha_loss = F.nll_loss(torch.log(ha_prob), target)  # Loss for heavy-antigen alignment
        hl_loss = F.nll_loss(torch.log(hl_prob), target)  # Loss for light-antigen alignment
        
        # Total loss is the sum of both losses
        total_loss = ha_loss + hl_loss
        
        return total_loss



