import torch
import torch.nn as nn
import torch.nn.functional as F

from diffab.modules.common.geometry import angstrom_to_nm, pairwise_dihedrals
from diffab.modules.common.layers import AngularEncoding
from diffab.utils.protein.constants import BBHeavyAtom, AA,max_num_heavyatoms
from diffab.modules.common.geometry import construct_3d_basis, global_to_local, get_backbone_dihedral_angles
from diffab.modules.encoders.ga import GAEncoder
from diffab.modules.common.so3 import rotation_to_so3vec

class PairEmbedding(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22, max_relpos=32):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.max_relpos = max_relpos
        self.aa_pair_embed = nn.Embedding(self.max_aa_types*self.max_aa_types, feat_dim)
        self.relpos_embed = nn.Embedding(2*max_relpos+1, feat_dim)

        self.aapair_to_distcoef = nn.Embedding(self.max_aa_types*self.max_aa_types, max_num_atoms*max_num_atoms)
        nn.init.zeros_(self.aapair_to_distcoef.weight)
        self.distance_embed = nn.Sequential(
            nn.Linear(max_num_atoms*max_num_atoms, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
        )

        self.dihedral_embed = AngularEncoding()
        feat_dihed_dim = self.dihedral_embed.get_out_dim(2)  # Phi and Psi

        infeat_dim = feat_dim + feat_dim + feat_dim + feat_dihed_dim
        self.out_mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, aa, res_nb, pos_atoms, mask_atoms, structure_mask=None, sequence_mask=None):
        """
        Args:
            aa: (N, L).
            res_nb: (N, L).
            pos_atoms:  (N, L, A, 3)
            mask_atoms: (N, L, A)
            structure_mask: (N, L)
            sequence_mask:  (N, L), mask out unknown amino acids to generate.

        Returns:
            (N, L, L, feat_dim)
        """
        N, L = aa.size()

        # Remove other atoms
        pos_atoms = pos_atoms[:, :, :self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]

        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]  # (N, L)
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :]
        pair_structure_mask = structure_mask[:, :, None] * structure_mask[:, None, :] if structure_mask is not None else None

        # Pair identities
        if sequence_mask is not None:
            # Avoid data leakage at training time
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))
        aa_pair = aa[:, :, None] * self.max_aa_types + aa[:, None, :]  # (N, L, L)
        feat_aapair = self.aa_pair_embed(aa_pair)

        # Relative sequential positions
        relpos = torch.clamp(
            res_nb[:, :, None] - res_nb[:, None, :],
            min=-self.max_relpos, max=self.max_relpos,
        )  # (N, L, L)
        feat_relpos = self.relpos_embed(relpos + self.max_relpos)

        # Distances
        d = angstrom_to_nm(torch.linalg.norm(
            pos_atoms[:, :, None, :, None] - pos_atoms[:, None, :, None, :],
            dim=-1, ord=2,
        )).reshape(N, L, L, -1)  # (N, L, L, A*A)
        c = F.softplus(self.aapair_to_distcoef(aa_pair))  # (N, L, L, A*A)
        d_gauss = torch.exp(-1 * c * d**2)
        mask_atom_pair = (mask_atoms[:, :, None, :, None] * mask_atoms[:, None, :, None, :]).reshape(N, L, L, -1)
        feat_dist = self.distance_embed(d_gauss * mask_atom_pair)
        if pair_structure_mask is not None:
            # Avoid data leakage at training time
            feat_dist = feat_dist * pair_structure_mask[:, :, :, None]

        # Orientations
        dihed = pairwise_dihedrals(pos_atoms)  # (N, L, L, 2)
        feat_dihed = self.dihedral_embed(dihed)
        if pair_structure_mask is not None:
            # Avoid data leakage at training time
            feat_dihed = feat_dihed * pair_structure_mask[:, :, :, None]

        # All
        feat_all = torch.cat([feat_aapair, feat_relpos, feat_dist, feat_dihed], dim=-1)
        feat_all = self.out_mlp(feat_all)  # (N, L, L, F)
        feat_all = feat_all * mask_pair[:, :, :, None]

        return feat_all
class ResidueEmbedding(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.aatype_embed = nn.Embedding(self.max_aa_types, feat_dim)
        self.dihed_embed = AngularEncoding()

        # 调整后的 infeat_dim 不再包含 fragment_type 的特征
        infeat_dim = feat_dim + (self.max_aa_types * max_num_atoms * 3) + self.dihed_embed.get_out_dim(3)
        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim * 2), nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, aa, res_nb, pos_atoms, mask_atoms, structure_mask=None, sequence_mask=None):
        """
        Args:
            aa: (N, L).
            res_nb: (N, L).
            pos_atoms: (N, L, A, 3).
            mask_atoms: (N, L, A).
            structure_mask: (N, L), mask out unknown structures to generate.
            sequence_mask: (N, L), mask out unknown amino acids to generate.
        """
        N, L = aa.size()
        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]  # (N, L)

        # Remove other atoms
        pos_atoms = pos_atoms[:, :, :self.max_num_atoms]
        mask_atoms = mask_atoms[:, :, :self.max_num_atoms]

        # Manually create a chain_nb assuming all residues are in the same chain
        chain_nb = torch.zeros_like(aa)  # (N, L), all residues belong to chain 0

        # Amino acid identity features
        if sequence_mask is not None:
            # Avoid data leakage at training time
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))
        aa_feat = self.aatype_embed(aa)  # (N, L, feat)

        # Coordinate features
        R = construct_3d_basis(
            pos_atoms[:, :, BBHeavyAtom.CA],
            pos_atoms[:, :, BBHeavyAtom.C],
            pos_atoms[:, :, BBHeavyAtom.N]
        )
        t = pos_atoms[:, :, BBHeavyAtom.CA]
        crd = global_to_local(R, t, pos_atoms)  # (N, L, A, 3)
        crd_mask = mask_atoms[:, :, :, None].expand_as(crd)
        crd = torch.where(crd_mask, crd, torch.zeros_like(crd))

        aa_expand = aa[:, :, None, None, None].expand(N, L, self.max_aa_types, self.max_num_atoms, 3)
        rng_expand = torch.arange(0, self.max_aa_types)[None, None, :, None, None].expand(N, L, self.max_aa_types, self.max_num_atoms, 3).to(aa_expand)
        place_mask = (aa_expand == rng_expand)
        crd_expand = crd[:, :, None, :, :].expand(N, L, self.max_aa_types, self.max_num_atoms, 3)
        crd_expand = torch.where(place_mask, crd_expand, torch.zeros_like(crd_expand))
        crd_feat = crd_expand.reshape(N, L, self.max_aa_types * self.max_num_atoms * 3)
        if structure_mask is not None:
            # Avoid data leakage at training time
            crd_feat = crd_feat * structure_mask[:, :, None]

        # Backbone dihedral features
        bb_dihedral, mask_bb_dihed = get_backbone_dihedral_angles(pos_atoms, chain_nb=chain_nb, res_nb=res_nb, mask=mask_residue)
        dihed_feat = self.dihed_embed(bb_dihedral[:, :, :, None]) * mask_bb_dihed[:, :, :, None]  # (N, L, 3, dihed/3)
        dihed_feat = dihed_feat.reshape(N, L, -1)
        if structure_mask is not None:
            # Avoid data leakage at training time
            dihed_mask = torch.logical_and(
                structure_mask,
                torch.logical_and(
                    torch.roll(structure_mask, shifts=+1, dims=1),
                    torch.roll(structure_mask, shifts=-1, dims=1)
                ),
            )  # Avoid slight data leakage via dihedral angles of anchor residues
            dihed_feat = dihed_feat * dihed_mask[:, :, None]

        # 最终的特征整合，移除了 type_feat
        out_feat = self.mlp(torch.cat([aa_feat, crd_feat, dihed_feat], dim=-1))  # (N, L, F)
        out_feat = out_feat * mask_residue[:, :, None]
        return out_feat
resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}
class diffabencoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.res_feat_dim = cfg.res_feat_dim

        num_atoms = resolution_to_num_atoms[cfg.get('resolution', 'full')]
        self.residue_embed = ResidueEmbedding(cfg.res_feat_dim, num_atoms)
        self.pair_embed = PairEmbedding(cfg.pair_feat_dim, num_atoms)
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(cfg.res_feat_dim * 2, cfg.res_feat_dim), nn.ReLU(),
            nn.Linear(cfg.res_feat_dim, cfg.res_feat_dim),
        )
        self.current_sequence_embedding = nn.Embedding(25, cfg.res_feat_dim)  # 22 is padding
        self.encoder = GAEncoder(cfg.res_feat_dim, cfg.pair_feat_dim, num_layers=6)

        # Adding adaptive average pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool1d(100)

    def encode(self, batch):
        """
        Returns:
            res_feat:   (N, L, res_feat_dim)
            pair_feat:  (N, L, L, pair_feat_dim)
        """
        structure_mask = None
        sequence_mask = None

        res_feat = self.residue_embed(
            aa=batch['aa'],
            res_nb=batch['res_nb'],

            pos_atoms=batch['pos_heavyatom'],
            mask_atoms=batch['mask_heavyatom'],
            structure_mask=structure_mask,
            sequence_mask=sequence_mask,
        )

        pair_feat = self.pair_embed(
            aa=batch['aa'],
            res_nb=batch['res_nb'],

            pos_atoms=batch['pos_heavyatom'],
            mask_atoms=batch['mask_heavyatom'],
            structure_mask=structure_mask,
            sequence_mask=sequence_mask,
        )

        R = construct_3d_basis(
            batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
        )
        p = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]

        return res_feat, pair_feat, R, p

    def forward(self, batch):
        mask_res = batch['mask']
        # N, L = batch['aa'].size()
        # mask_res = torch.ones(N, L, device=batch['aa'].device, dtype=batch['aa'].dtype)  # 不进行掩码
        res_feat, pair_feat, R_0, p_0 = self.encode(batch)
        v_0 = rotation_to_so3vec(R_0)
        s_0 = batch['aa']
        res_feat = self.res_feat_mixer(torch.cat([res_feat, self.current_sequence_embedding(s_0)], dim=-1))  # [Important] Incorporate sequence at the current step.
        res_feat = self.encoder(R_0, p_0, res_feat, pair_feat, mask_res)

        # Apply adaptive average pooling to pool the length to 100
        res_feat = res_feat.permute(0, 2, 1)  # Change shape to (N, res_feat_dim, L) for pooling
        res_feat = self.adaptive_pool(res_feat)  # Apply pooling
        res_feat = res_feat.permute(0, 2, 1)  # Change shape back to (N, 100, res_feat_dim)

        return res_feat

class EZPairEmbedding(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22, max_relpos=32):  # 更进一步缩小参数
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.max_relpos = max_relpos
        
        # 保留基本的氨基酸和相对位置嵌入
        self.aa_pair_embed = nn.Embedding(self.max_aa_types * self.max_aa_types, feat_dim)
        self.relpos_embed = nn.Embedding(2 * max_relpos + 1, feat_dim)

        infeat_dim = feat_dim * 2  # 相对位置和氨基酸对嵌入的特征维度
        self.out_mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, aa, res_nb, pos_atoms, mask_atoms, structure_mask=None, sequence_mask=None):
        N, L = aa.size()

        # Remove other atoms, directly use CA atom, reducing computational complexity
        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]
        mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :]

        # Pair identities
        if sequence_mask is not None:
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))
        aa_pair = aa[:, :, None] * self.max_aa_types + aa[:, None, :]
        feat_aapair = self.aa_pair_embed(aa_pair)

        # Relative sequential positions
        relpos = torch.clamp(
            res_nb[:, :, None] - res_nb[:, None, :],
            min=-self.max_relpos, max=self.max_relpos,
        )
        feat_relpos = self.relpos_embed(relpos + self.max_relpos)

        # Only keeping amino acid pair and relative position embeddings
        feat_all = torch.cat([feat_aapair, feat_relpos], dim=-1)
        feat_all = self.out_mlp(feat_all)
        feat_all = feat_all * mask_pair[:, :, :, None]

        return feat_all


class EZResidueEmbedding(nn.Module):

    def __init__(self, feat_dim, max_num_atoms, max_aa_types=22):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.aatype_embed = nn.Embedding(self.max_aa_types, feat_dim)

        infeat_dim = feat_dim  # 只有氨基酸特征的嵌入
        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, aa, res_nb, pos_atoms, mask_atoms, structure_mask=None, sequence_mask=None):
        N, L = aa.size()
        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]

        # Amino acid identity features
        if sequence_mask is not None:
            aa = torch.where(sequence_mask, aa, torch.full_like(aa, fill_value=AA.UNK))
        aa_feat = self.aatype_embed(aa)

        # Directly pass through amino acid embedding without 3D positional or angular encoding
        out_feat = self.mlp(aa_feat)
        out_feat = out_feat * mask_residue[:, :, None]
        return out_feat


class mlpencoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.res_feat_dim = cfg.res_feat_dim

        num_atoms = resolution_to_num_atoms[cfg.get('resolution', 'full')]
        self.residue_embed = EZResidueEmbedding(cfg.res_feat_dim, num_atoms)
        self.pair_embed = EZPairEmbedding(cfg.pair_feat_dim, num_atoms)
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(cfg.res_feat_dim * 2, cfg.res_feat_dim), nn.ReLU(),
            nn.Linear(cfg.res_feat_dim, cfg.res_feat_dim),
        )
        self.current_sequence_embedding = nn.Embedding(25, cfg.res_feat_dim)

        self.simple_encoder = nn.Sequential(
            nn.Linear(cfg.res_feat_dim, cfg.res_feat_dim),
            nn.ReLU(),
            nn.Linear(cfg.res_feat_dim, cfg.res_feat_dim),
            nn.ReLU(),
            nn.Linear(cfg.res_feat_dim, cfg.res_feat_dim),
        )

        # Adding an adaptive pooling layer to get the output of shape (N, 100, feat_dim)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(100)  # Change to nn.AdaptiveMaxPool1d if needed

    def encode(self, batch):
        structure_mask = None
        sequence_mask = None

        res_feat = self.residue_embed(
            aa=batch['aa'],
            res_nb=batch['res_nb'],
            pos_atoms=batch['pos_heavyatom'],
            mask_atoms=batch['mask_heavyatom'],
            structure_mask=structure_mask,
            sequence_mask=sequence_mask,
        )

        pair_feat = self.pair_embed(
            aa=batch['aa'],
            res_nb=batch['res_nb'],
            pos_atoms=batch['pos_heavyatom'],
            mask_atoms=batch['mask_heavyatom'],
            structure_mask=structure_mask,
            sequence_mask=sequence_mask,
        )

        R = construct_3d_basis(
            batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.C],
            batch['pos_heavyatom'][:, :, BBHeavyAtom.N],
        )
        p = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]

        return res_feat, pair_feat, R, p

    def forward(self, batch):
        mask_res = batch['mask']
        res_feat, pair_feat, R_0, p_0 = self.encode(batch)
        s_0 = batch['aa']
        res_feat = self.res_feat_mixer(torch.cat([res_feat, self.current_sequence_embedding(s_0)], dim=-1))

        res_feat = self.simple_encoder(res_feat)

        # Transpose to (N, feat_dim, L) for adaptive pooling
        res_feat = res_feat.permute(0, 2, 1)

        # Apply adaptive pooling to get (N, feat_dim, 100)
        res_feat = self.adaptive_pool(res_feat)

        # Transpose back to (N, 100, feat_dim)
        res_feat = res_feat.permute(0, 2, 1)

        return res_feat

