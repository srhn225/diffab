import torch
import torch.nn as nn

from diffab.modules.common.geometry import construct_3d_basis
from diffab.modules.common.so3 import rotation_to_so3vec
from encoder.embedding import *
from diffab.utils.protein.constants import max_num_heavyatoms, BBHeavyAtom

def create_contrastive_masks(batch_size, device='cpu'):
    """
    Create positive and negative masks for contrastive learning.
    
    Args:
        batch_size (int): The number of samples in the batch.
        device (str): The device for tensor allocation ('cpu' or 'cuda').
    
    Returns:
        pos_mask (Tensor): Positive mask of shape (batch_size, batch_size)
        neg_mask (Tensor): Negative mask of shape (batch_size, batch_size)
    """
    # Create an identity matrix for the positive mask
    pos_mask = torch.eye(batch_size, device=device)  # Shape: (batch_size, batch_size)
    
    # The negative mask is the inverse of the positive mask
    neg_mask = 1 - pos_mask  # Shape: (batch_size, batch_size)
    
    return pos_mask, neg_mask

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveDiffAb(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Two diffabencoders: one for the heavy-light pair and one for the antigen
        self.heavy_encoder = diffabencoder(cfg)
        self.light_encoder = diffabencoder(cfg)
        self.antigen_encoder = diffabencoder(cfg)
        
    def forward(self, batch):
        # Get batch size and device
        batch_size = batch['heavy']['aa'].size(0)
        device = batch['heavy']['aa'].device
        
        # Create contrastive masks
        pos_mask, neg_mask = create_contrastive_masks(batch_size, device)
        
        # Encode the heavy and light chains separately
        heavy_feat = self.heavy_encoder(batch['heavy'])  # Shape: (N, L, res_feat_dim)
        light_feat = self.light_encoder(batch['light'])  # Shape: (N, L, res_feat_dim)

        # Encode the antigen
        antigen_feat = self.antigen_encoder(batch['antigen'])  # Shape: (N, L, res_feat_dim)
        
        # Compute pairwise dot product between heavy chain and antigen
        heavy_dot_product = torch.bmm(heavy_feat, antigen_feat.transpose(1, 2))  # Shape: (N, L, L)
        
        # Compute pairwise dot product between light chain and antigen
        light_dot_product = torch.bmm(light_feat, antigen_feat.transpose(1, 2))  # Shape: (N, L, L)
        
        # Contrastive loss: maximize dot product for paired samples and minimize for unpaired
        # Note: we apply masks along batch dimension, so we need to sum over all dimensions
        heavy_positive_loss = -torch.sum(pos_mask * heavy_dot_product) / (pos_mask.sum() + 1e-6)
        light_positive_loss = -torch.sum(pos_mask * light_dot_product) / (pos_mask.sum() + 1e-6)
        
        heavy_negative_loss = torch.sum(neg_mask * heavy_dot_product) / (neg_mask.sum() + 1e-6)
        light_negative_loss = torch.sum(neg_mask * light_dot_product) / (neg_mask.sum() + 1e-6)
        
        # Total loss: combine heavy and light chain losses
        total_loss = heavy_positive_loss + light_positive_loss + heavy_negative_loss + light_negative_loss
        return total_loss



