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
        
        # 根据 cfg 中的 encoder 名称动态选择编码器
        type = cfg.get('type', 'simple')  # 默认为 'simple'，如果没有配置则使用简单编码器
        if type == 'diffab':
            self.encoder_class = diffabencoder
            print("using diffab encoder")
        else:
            self.encoder_class = mlpencoder  # 默认使用实现的 simple encoder
            print("using baseline encoder")        
        # 初始化编码器
        self.heavy_encoder = self.encoder_class(cfg)
        self.light_encoder = self.encoder_class(cfg)
        self.antigen_encoder = self.encoder_class(cfg)
        self.temperature = 0.1  # 可以调节的超参数
    
    def forward(self, batch):
        # 获取 batch 的大小和设备
        batch_size = batch['heavy']['aa'].size(0)
        device = batch['heavy']['aa'].device
        
        # 编码 heavy 链、light 链和 antigen
        heavy_feat = self.heavy_encoder(batch['heavy'])  # 输出形状: (N, L)
        light_feat = self.light_encoder(batch['light'])  # 输出形状: (N, L)
        antigen_feat = self.antigen_encoder(batch['antigen'])  # 输出形状: (N, L)
        
        # 对 antigen 特征进行转置以便进行矩阵乘法
        transposed_antigen = torch.transpose(antigen_feat, 0, 1)  # 输出形状: (L, N)
        
        # 计算点积
        ha_dot = torch.matmul(heavy_feat, transposed_antigen)  # 输出形状: (N, N)
        hl_dot = torch.matmul(light_feat, transposed_antigen)  # 输出形状: (N, N)
        
        # 对每一行应用 softmax 获得概率
        ha_prob = F.softmax(ha_dot, dim=1)  # 输出形状: (N, N)
        hl_prob = F.softmax(hl_dot, dim=1)  # 输出形状: (N, N)
        
        # 创建目标标签，目标是对角线元素的索引
        target = torch.arange(batch_size, device=device)  # 输出形状: (N,)
        
        # 计算负对数似然损失
        ha_loss = F.nll_loss(torch.log(ha_prob), target)  # Heavy 链与 antigen 对齐的损失
        hl_loss = F.nll_loss(torch.log(hl_prob), target)  # Light 链与 antigen 对齐的损失
        
        # 总损失是两个损失的和
        total_loss = ha_loss + hl_loss
        
        return total_loss




