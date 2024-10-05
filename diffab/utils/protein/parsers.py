import torch
from Bio.PDB import Selection,NeighborSearch
from Bio.PDB.Residue import Residue
from easydict import EasyDict


from .constants import (
    AA, max_num_heavyatoms,
    restype_to_heavyatom_names, 
    BBHeavyAtom
)
import numpy as np

class ParsingException(Exception):
    pass


def _get_residue_heavyatom_info(res: Residue):
    pos_heavyatom = torch.zeros([max_num_heavyatoms, 3], dtype=torch.float)
    mask_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.bool)
    restype = AA(res.get_resname())
    for idx, atom_name in enumerate(restype_to_heavyatom_names[restype]):
        if atom_name == '': continue
        if atom_name in res:
            pos_heavyatom[idx] = torch.tensor(res[atom_name].get_coord().tolist(), dtype=pos_heavyatom.dtype)
            mask_heavyatom[idx] = True
    return pos_heavyatom, mask_heavyatom


def parse_biopython_structure(entity, unknown_threshold=1.0, max_resseq=None):
    chains = Selection.unfold_entities(entity, 'C')
    chains.sort(key=lambda c: c.get_id())
    data = EasyDict({
        'chain_id': [],
        'resseq': [], 'icode': [], 'res_nb': [],
        'aa': [],
        'pos_heavyatom': [], 'mask_heavyatom': [],
    })
    tensor_types = {
        'resseq': torch.LongTensor,
        'res_nb': torch.LongTensor,
        'aa': torch.LongTensor,
        'pos_heavyatom': torch.stack,
        'mask_heavyatom': torch.stack,
    }

    count_aa, count_unk = 0, 0

    for i, chain in enumerate(chains):
        seq_this = 0   # Renumbering residues
        residues = Selection.unfold_entities(chain, 'R')
        residues.sort(key=lambda res: (res.get_id()[1], res.get_id()[2]))   # Sort residues by resseq-icode
        for _, res in enumerate(residues):
            resseq_this = int(res.get_id()[1])
            if max_resseq is not None and resseq_this > max_resseq:
                continue

            resname = res.get_resname()
            if not AA.is_aa(resname): continue
            if not (res.has_id('CA') and res.has_id('C') and res.has_id('N')): continue
            restype = AA(resname)
            count_aa += 1
            if restype == AA.UNK: 
                count_unk += 1
                continue

            # Chain info
            data.chain_id.append(chain.get_id())

            # Residue types
            data.aa.append(restype) # Will be automatically cast to torch.long

            # Heavy atoms
            pos_heavyatom, mask_heavyatom = _get_residue_heavyatom_info(res)
            data.pos_heavyatom.append(pos_heavyatom)
            data.mask_heavyatom.append(mask_heavyatom)

            # Sequential number
            resseq_this = int(res.get_id()[1])
            icode_this = res.get_id()[2]
            if seq_this == 0:
                seq_this = 1
            else:
                d_CA_CA = torch.linalg.norm(data.pos_heavyatom[-2][BBHeavyAtom.CA] - data.pos_heavyatom[-1][BBHeavyAtom.CA], ord=2).item()
                if d_CA_CA <= 4.0:
                    seq_this += 1
                else:
                    d_resseq = resseq_this - data.resseq[-1]
                    seq_this += max(2, d_resseq)

            data.resseq.append(resseq_this)
            data.icode.append(icode_this)
            data.res_nb.append(seq_this)

    if len(data.aa) == 0:
        raise ParsingException('No parsed residues.')

    if (count_unk / count_aa) >= unknown_threshold:
        raise ParsingException(
            f'Too many unknown residues, threshold {unknown_threshold:.2f}.'
        )

    seq_map = {}
    for i, (chain_id, resseq, icode) in enumerate(zip(data.chain_id, data.resseq, data.icode)):
        seq_map[(chain_id, resseq, icode)] = i

    for key, convert_fn in tensor_types.items():
        data[key] = convert_fn(data[key])

    return data, seq_map
def parse_biopython_structure_antigen(entity, antibody_atoms=None, unknown_threshold=1.0, max_resseq=None):
    chains = Selection.unfold_entities(entity, 'C')
    chains.sort(key=lambda c: c.get_id())
    
    data = EasyDict({
        'chain_id': [],
        'resseq': [], 'icode': [], 'res_nb': [],
        'aa': [],
        'pos_heavyatom': [], 'mask_heavyatom': [],
    })
    
    tensor_types = {
        'resseq': torch.LongTensor,
        'res_nb': torch.LongTensor,
        'aa': torch.LongTensor,
        'pos_heavyatom': torch.stack,
        'mask_heavyatom': torch.stack,
    }
    
    count_aa, count_unk = 0, 0
    contact_mask = []  # 用于记录是否接触抗体
    ns = None
    
    # 如果提供了抗体原子信息，则初始化NeighborSearch
    if antibody_atoms:
        ns = NeighborSearch(antibody_atoms)
    
    for i, chain in enumerate(chains):
        seq_this = 0  # Renumbering residues
        residues = Selection.unfold_entities(chain, 'R')
        residues.sort(key=lambda res: (res.get_id()[1], res.get_id()[2]))  # Sort residues by resseq-icode
        
        for _, res in enumerate(residues):
            resseq_this = int(res.get_id()[1])
            if max_resseq is not None and resseq_this > max_resseq:
                continue

            resname = res.get_resname()
            if not AA.is_aa(resname): continue
            if not (res.has_id('CA') and res.has_id('C') and res.has_id('N')): continue
            
            restype = AA(resname)
            count_aa += 1
            if restype == AA.UNK: 
                count_unk += 1
                continue

            # 抗原接触检测：如果提供了抗体信息，计算该抗原残基是否与抗体CDR接触
            if ns:
                antigen_atom_positions = np.array([atom.get_coord() for atom in res.get_atoms()])
                average_position = np.mean(antigen_atom_positions, axis=0)
                close_atoms = ns.search(average_position, 10.0)
                
                # 如果没有与抗体CDR接触，则跳过该残基
                if len(close_atoms) == 0:
                    continue
            
            # Chain info
            data.chain_id.append(chain.get_id())

            # Residue types
            data.aa.append(restype)  # Will be automatically cast to torch.long

            # Heavy atoms
            pos_heavyatom, mask_heavyatom = _get_residue_heavyatom_info(res)
            data.pos_heavyatom.append(pos_heavyatom)
            data.mask_heavyatom.append(mask_heavyatom)

            # Sequential number
            resseq_this = int(res.get_id()[1])
            icode_this = res.get_id()[2]
            if seq_this == 0:
                seq_this = 1
            else:
                d_CA_CA = torch.linalg.norm(data.pos_heavyatom[-2][BBHeavyAtom.CA] - data.pos_heavyatom[-1][BBHeavyAtom.CA], ord=2).item()
                if d_CA_CA <= 4.0:
                    seq_this += 1
                else:
                    d_resseq = resseq_this - data.resseq[-1]
                    seq_this += max(2, d_resseq)

            data.resseq.append(resseq_this)
            data.icode.append(icode_this)
            data.res_nb.append(seq_this)
    
    # 检查是否有残基被解析
    if len(data.aa) == 0:
        raise ParsingException('No parsed residues.')
    
    # 检查未知残基是否超出阈值
    if (count_unk / count_aa) >= unknown_threshold:
        raise ParsingException(
            f'Too many unknown residues, threshold {unknown_threshold:.2f}.'
        )
    
    # 生成序列映射
    seq_map = {}
    for i, (chain_id, resseq, icode) in enumerate(zip(data.chain_id, data.resseq, data.icode)):
        seq_map[(chain_id, resseq, icode)] = i
    
    # 转换数据为Tensor格式
    for key, convert_fn in tensor_types.items():
        data[key] = convert_fn(data[key])
    
    return data, seq_map
