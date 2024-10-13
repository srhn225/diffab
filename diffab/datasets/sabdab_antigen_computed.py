import os
import random
import logging
import datetime
import pandas as pd
import joblib
import pickle
import lmdb
import subprocess
import torch
from Bio import PDB, SeqRecord, SeqIO, Seq
from Bio.PDB import PDBExceptions
from Bio.PDB import Polypeptide
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from Bio.PDB import NeighborSearch
from ..utils.protein import parsers, constants
from ._base import register_dataset
import numpy as np
from encoder.models import ContrastiveDiffAb
from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.train import *
import torch
import torch.nn.functional as F
from encoder.test.choose import load_tensors_from_directory,find_top_similar
ALLOWED_AG_TYPES = {
    'protein',
    'protein | protein',
    'protein | protein | protein',
    'protein | protein | protein | protein | protein',
    'protein | protein | protein | protein',
}

RESOLUTION_THRESHOLD = 4.0

TEST_ANTIGENS = [
    'sars-cov-2 receptor binding domain',
    'hiv-1 envelope glycoprotein gp160',
    'mers s',
    'influenza a virus',
    'cd27 antigen',
]


def nan_to_empty_string(val):
    if val != val or not val:
        return ''
    else:
        return val


def nan_to_none(val):
    if val != val or not val:
        return None
    else:
        return val


def split_sabdab_delimited_str(val):
    if not val:
        return []
    else:
        return [s.strip() for s in val.split('|')]


def parse_sabdab_resolution(val):
    if val == 'NOT' or not val or val != val:
        return None
    elif isinstance(val, str) and ',' in val:
        return float(val.split(',')[0].strip())
    else:
        return float(val)


def _aa_tensor_to_sequence(aa):
    return ''.join([Polypeptide.index_to_one(a.item()) for a in aa.flatten()])


def _label_heavy_chain_cdr(data, seq_map, max_cdr3_length=30):
    if data is None or seq_map is None:
        return data, seq_map

    # Add CDR labels
    cdr_flag = torch.zeros_like(data['aa'])
    for position, idx in seq_map.items():
        resseq = position[1]
        cdr_type = constants.ChothiaCDRRange.to_cdr('H', resseq)
        if cdr_type is not None:
            cdr_flag[idx] = cdr_type
    data['cdr_flag'] = cdr_flag

    # Add CDR sequence annotations
    data['H1_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.H1] )
    data['H2_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.H2] )
    data['H3_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.H3] )

    cdr3_length = (cdr_flag == constants.CDR.H3).sum().item()
    # Remove too long CDR3
    if cdr3_length > max_cdr3_length:
        cdr_flag[cdr_flag == constants.CDR.H3] = 0
        logging.warning(f'CDR-H3 too long {cdr3_length}. Removed.')
        return None, None

    # Filter: ensure CDR3 exists
    if cdr3_length == 0:
        logging.warning('No CDR-H3 found in the heavy chain.')
        return None, None

    return data, seq_map


def _label_light_chain_cdr(data, seq_map, max_cdr3_length=30):
    if data is None or seq_map is None:
        return data, seq_map
    cdr_flag = torch.zeros_like(data['aa'])
    for position, idx in seq_map.items():
        resseq = position[1]
        cdr_type = constants.ChothiaCDRRange.to_cdr('L', resseq)
        if cdr_type is not None:
            cdr_flag[idx] = cdr_type
    data['cdr_flag'] = cdr_flag

    data['L1_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.L1] )
    data['L2_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.L2] )
    data['L3_seq'] = _aa_tensor_to_sequence( data['aa'][cdr_flag == constants.CDR.L3] )

    cdr3_length = (cdr_flag == constants.CDR.L3).sum().item()
    # Remove too long CDR3
    if cdr3_length > max_cdr3_length:
        cdr_flag[cdr_flag == constants.CDR.L3] = 0
        logging.warning(f'CDR-L3 too long {cdr3_length}. Removed.')
        return None, None

    # Ensure CDR3 exists
    if cdr3_length == 0:
        logging.warning('No CDRs found in the light chain.')
        return None, None

    return data, seq_map

def process_dict(input_dict):# 将一条数据包装一下，假装是一个batch
    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            # 为 tensor 增加一个维度
            input_dict[key] = value.unsqueeze(0)
        elif isinstance(value, list):
            # 将 list 转换为竖着的 tuple
            input_dict[key] = [(item,) for item in value]
    return input_dict
def preprocess_sabdab_structure(task):
    entry = task['entry']
    pdb_path = task['pdb_path']

    parser = PDB.PDBParser(QUIET=True)
    model = parser.get_structure(id, pdb_path)[0]


    parsed = {
        'id': entry['id'],
        'heavy': None,
        'heavy_seqmap': None,
        'light': None,
        'light_seqmap': None,
        'antigen': None,
        'antigen_seqmap': None,
    }

    try:
        # 检查重链和轻链的存在性
        if entry['H_chain'] is None or entry['L_chain'] is None:
            raise ValueError('Missing both heavy and light chains.')
        # 检查抗原链是否存在
        if len(entry['ag_chains']) == 0:
            raise ValueError('No antigen chains found.')

        if entry['H_chain'] is not None:
            (
                parsed['heavy'], 
                parsed['heavy_seqmap']
            ) = _label_heavy_chain_cdr(*parsers.parse_biopython_structure(
                model[entry['H_chain']],
                max_resseq = 113    # Chothia, end of Heavy chain Fv
            ))
        
        if entry['L_chain'] is not None:
            (
                parsed['light'], 
                parsed['light_seqmap']
            ) = _label_light_chain_cdr(*parsers.parse_biopython_structure(
                model[entry['L_chain']],
                max_resseq = 106    # Chothia, end of Light chain Fv
            ))
        if parsed['heavy'] is None and parsed['light'] is None:
            raise ValueError('Neither valid H-chaincdr or L-chaincdr is found.')       
        if len(entry['ag_chains']) > 0:
            chains = [model[c] for c in entry['ag_chains']]
            (
                parsed['antigen'], 
                parsed['antigen_seqmap']
            ) = parsers.parse_biopython_structure(chains)

        # 处理抗原链
        if len(entry['ag_chains']) > 0:
            chains = [model[c] for c in entry['ag_chains']]


            # 获取所有抗体（重链和轻链）的原子用于邻近搜索
            antibody_atoms = []
            if parsed['heavy'] is not None:
                for heavy_res in model[entry['H_chain']].get_residues():
                    antibody_atoms.extend(heavy_res.get_atoms())
            if parsed['light'] is not None:
                for light_res in model[entry['L_chain']].get_residues():
                    antibody_atoms.extend(light_res.get_atoms())
            antigen_truncated, antigen_seqmap = parsers.parse_biopython_structure_antigen(chains,antibody_atoms=antibody_atoms)
            # print(antigen_truncated)
            # for key in antigen_truncated:
            #     print(key+type(antigen_truncated[key]).__name__)

            # batch={}
            # # 遍历antigen_truncated的每个key-value
            # for key, value in antigen_truncated.items():
            #     # 如果value已经是tensor或类似数组的形式，添加一个batch维度
            #     # 比如，如果value是 (n,) 的向量，变为 (1, n)
            #     batch[key] = torch.unsqueeze(torch.tensor(value), dim=0)
            diffabmodel=task['model']
            # print(batch)
            antigen_truncated['mask']=torch.ones([antigen_truncated['aa'].size(0)], dtype=torch.bool)
            antigen_batchlike=recursive_to(process_dict(antigen_truncated),'cuda')



            antigen_feature=diffabmodel.compute_antigen_feature(antigen_batchlike)

            # print(antigen_feature)
            parsed['antigen_feature']=antigen_feature
            feature_all=task['feature_all']
            feature_all_device=recursive_to(feature_all,'cuda')
            similar_heavy,similar_light=compute_cosine_similarity(feature_all_device,antigen_feature[0])
            top10_heavy=find_top_similar(similar_heavy,id=task['id'])
            top10_light=find_top_similar(similar_light,id=task['id'])#id用于去除掉自己，防止出现数据泄露
            # print(top10_heavy)
            # print(top10_light)

            parsed['topn_heavy']=top10_heavy
            parsed['topn_light']=top10_light


            # # 使用NeighborSearch进行抗原链-抗体链原子接触搜索
            # ns = NeighborSearch(antibody_atoms)

            # # 生成抗原链的布尔掩码，判断每个残基是否与抗体链的原子距离小于10Å
            # contact_mask = []
            # for chain in chains:
            #     chain_contacts = []
            #     for antigen_res in chain.get_residues():
            #         antigen_atom_positions = np.array([atom.get_coord() for atom in antigen_res.get_atoms()])

            #         # 计算坐标的平均值
            #         average_position = np.mean(antigen_atom_positions, axis=0)



            #         # 使用邻近搜索找出距离小于10Å的原子
            #         close_atoms = ns.search(average_position,10.0)
            #         # 如果有任意原子在10Å以内，标记为True，否则为False
            #         chain_contacts.append(len(close_atoms) > 0)
            #     contact_mask.append(torch.tensor(chain_contacts, dtype=torch.bool))

            # long_mask = torch.cat(contact_mask, dim=0)  # 保存接触掩码

            # seq_keys = list(antigen_seqmap.keys())


            # parsed['antigen_seqmap'] = {
            #     key: antigen_seqmap[key] 
            #     for i, key in enumerate(seq_keys) 
            #     if long_mask[i]
            # }

            # for key in antigen_residue:
            #     if isinstance(antigen_residue[key], torch.Tensor):
            #         antigen_residue[key] = antigen_residue[key][long_mask]  # 更新字典中的原始元素
            #     if isinstance(light_residues[key],list):
            #         antigen_residue[key] = [item for i, item in enumerate(antigen_residue[key]) if long_mask[i] > 0]
            # parsed['antigen']=antigen_residue            




    except (PDBExceptions.PDBConstructionException, parsers.ParsingException, KeyError, ValueError) as e:
        logging.warning(f"[{task['id']}] {e.__class__.__name__}: {str(e)}")
        return None
    
    return parsed





class SAbDabDatasetWithtopn(Dataset):

    MAP_SIZE = 32*(1024*1024*1024)  # 32GB

    def __init__(
        self, 
        model,
        summary_path = './data/sabdab_summary_all.tsv', 
        chothia_dir = './data/all_structures/chothia', 
        processed_dir = './data/processed',
        split = 'train',
        split_seed = 2022,
        transform = None,
        reset = False,
        filter_invalid = False,
        feature_all=None

    ):
        super().__init__()
        self.feature_all=feature_all
        self.model=model
        self.summary_path = summary_path
        self.filter_invalid=filter_invalid
        self.chothia_dir = chothia_dir
        if not os.path.exists(chothia_dir):
            raise FileNotFoundError(
                f"SAbDab structures not found in {chothia_dir}. "
                "Please download them from http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/"
            )
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)

        self.sabdab_entries = None
        self._load_sabdab_entries()

        self.db_conn = None
        self.db_ids = None
        self._load_structures(reset)

        self.clusters = None
        self.id_to_cluster = None
        self._load_clusters(reset)

        self.ids_in_split = None
        self._load_split(split, split_seed)

        self.transform = transform
                # 如果设置了filter_invalid，就过滤掉无效条目

    def filter_invalid_entries(self):
        """过滤掉heavy、light或antigen链为空的数据条目"""
        self._connect_db()  # 确保数据库已连接
        valid_entries = []
        for entry in self.sabdab_entries:
            structure = self.get_structure(entry['id'])
            if structure['heavy'] is not None and structure['light'] is not None and structure['antigen'] is not None:
                valid_entries.append(entry)
        print(f"总数量: {len(self.sabdab_entries)}")
        self.sabdab_entries = valid_entries
        print(f"过滤后剩余有效条目数量: {len(self.sabdab_entries)}")


    def _load_sabdab_entries(self):
        df = pd.read_csv(self.summary_path, sep='\t')
        entries_all = []
        for i, row in tqdm(
            df.iterrows(), 
            dynamic_ncols=True, 
            desc='Loading entries',
            total=len(df),
        ):
            entry_id = "{pdbcode}_{H}_{L}_{Ag}".format(
                pdbcode = row['pdb'],
                H = nan_to_empty_string(row['Hchain']),
                L = nan_to_empty_string(row['Lchain']),
                Ag = ''.join(split_sabdab_delimited_str(
                    nan_to_empty_string(row['antigen_chain'])
                ))
            )
            ag_chains = split_sabdab_delimited_str(
                nan_to_empty_string(row['antigen_chain'])
            )
            resolution = parse_sabdab_resolution(row['resolution'])
            entry = {
                'id': entry_id,
                'pdbcode': row['pdb'],
                'H_chain': nan_to_none(row['Hchain']),
                'L_chain': nan_to_none(row['Lchain']),
                'ag_chains': ag_chains,
                'ag_type': nan_to_none(row['antigen_type']),
                'ag_name': nan_to_none(row['antigen_name']),
                'date': datetime.datetime.strptime(row['date'], '%m/%d/%y'),
                'resolution': resolution,
                'method': row['method'],
                'scfv': row['scfv'],
            }

            # Filtering
            if (
                (entry['ag_type'] in ALLOWED_AG_TYPES or entry['ag_type'] is None)
                and (entry['resolution'] is not None and entry['resolution'] <= RESOLUTION_THRESHOLD)
            ):
                entries_all.append(entry)
        self.sabdab_entries = entries_all

    def _load_structures(self, reset):
        if not os.path.exists(self._structure_cache_path) or reset:
            if os.path.exists(self._structure_cache_path):
                os.unlink(self._structure_cache_path)
            self._preprocess_structures()

        with open(self._structure_cache_path + '-ids', 'rb') as f:
            self.db_ids = pickle.load(f)
        
        # 过滤无效的结构数据
        self.sabdab_entries = list(
            filter(
                lambda e: e['id'] in self.db_ids,
                self.sabdab_entries
            )
        )

        # 这里可以插入过滤无效条目的逻辑
        if self.filter_invalid:
            self.filter_invalid_entries()


    @property
    def _structure_cache_path(self):
        return os.path.join(self.processed_dir, 'structures-computed.lmdb')
        
    def _preprocess_structures(self):
        tasks = []
        for entry in self.sabdab_entries:
            pdb_path = os.path.join(self.chothia_dir, '{}.pdb'.format(entry['pdbcode']))
            if not os.path.exists(pdb_path):
                logging.warning(f"PDB not found: {pdb_path}")
                continue
            tasks.append({
                'id': entry['id'],
                'entry': entry,
                'pdb_path': pdb_path,
                'model': self.model,  # 仍然保留 model，但只是顺序执行
                'feature_all':self.feature_all
            })

        # 改为逐条处理任务
        data_list = []
        for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess'):
            data = preprocess_sabdab_structure(task)
            data_list.append(data)


        db_conn = lmdb.open(
            self._structure_cache_path,
            map_size = self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )
        ids = []
        with db_conn.begin(write=True, buffers=True) as txn:
            for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
                if data is None:
                    continue
                ids.append(data['id'])
                txn.put(data['id'].encode('utf-8'), pickle.dumps(data))

        with open(self._structure_cache_path + '-ids', 'wb') as f:
            pickle.dump(ids, f)

    @property
    def _cluster_path(self):
        return os.path.join(self.processed_dir, 'cluster_result_cluster.tsv')

    def _load_clusters(self, reset):
        if not os.path.exists(self._cluster_path) or reset:
            self._create_clusters()

        clusters, id_to_cluster = {}, {}
        with open(self._cluster_path, 'r') as f:
            for line in f.readlines():
                cluster_name, data_id = line.split()
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(data_id)
                id_to_cluster[data_id] = cluster_name
        self.clusters = clusters
        self.id_to_cluster = id_to_cluster

    def _create_clusters(self):
        cdr_records = []
        for id in self.db_ids:
            structure = self.get_structure(id)
            if structure['heavy'] is not None:
                cdr_records.append(SeqRecord.SeqRecord(
                    Seq.Seq(structure['heavy']['H3_seq']),
                    id = structure['id'],
                    name = '',
                    description = '',
                ))
            elif structure['light'] is not None:
                cdr_records.append(SeqRecord.SeqRecord(
                    Seq.Seq(structure['light']['L3_seq']),
                    id = structure['id'],
                    name = '',
                    description = '',
                ))
        fasta_path = os.path.join(self.processed_dir, 'cdr_sequences.fasta')
        SeqIO.write(cdr_records, fasta_path, 'fasta')

        cmd = ' '.join([
            'mmseqs', 'easy-cluster',
            os.path.realpath(fasta_path),
            'cluster_result', 'cluster_tmp',
            '--min-seq-id', '0.5',
            '-c', '0.8',
            '--cov-mode', '1',
        ])
        subprocess.run(cmd, cwd=self.processed_dir, shell=True, check=True)

    def _load_split(self, split, split_seed):
        assert split in ('train', 'val', 'test')
        ids_test = [
            entry['id']
            for entry in self.sabdab_entries
            if entry['ag_name'] in TEST_ANTIGENS
        ]
        test_relevant_clusters = set([self.id_to_cluster[id] for id in ids_test])

        ids_train_val = [
            entry['id']
            for entry in self.sabdab_entries
            if self.id_to_cluster[entry['id']] not in test_relevant_clusters
        ]
        random.Random(split_seed).shuffle(ids_train_val)
        if split == 'test':
            self.ids_in_split = ids_test
        elif split == 'val':
            self.ids_in_split = ids_train_val[:20]
        else:
            self.ids_in_split = ids_train_val[20:]

    def _connect_db(self):
        if self.db_conn is not None:
            return
        self.db_conn = lmdb.open(
            self._structure_cache_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def get_structure(self, id):
        self._connect_db()
        with self.db_conn.begin() as txn:
            return pickle.loads(txn.get(id.encode()))

    def __len__(self):
        return len(self.ids_in_split)

    def __getitem__(self, index):
        id = self.ids_in_split[index]
        data = self.get_structure(id)
        if self.transform is not None:
            data = self.transform(data)
        return data


@register_dataset('sabdabtopn')
def get_sabdab_dataset(cfg, transform):
    return SAbDabDatasetWithtopn(
        summary_path = cfg.summary_path,
        chothia_dir = cfg.chothia_dir,
        processed_dir = cfg.processed_dir,
        split = cfg.split,
        split_seed = cfg.get('split_seed', 2022),
        transform = transform,
        filter_invalid = cfg.get('filter', False),
        model=None
    )



def compute_cosine_similarity(data, target_tensor):
    """Compute cosine similarity between target_tensor and all tensors in data."""
    similarities_heavy = {}
    similarities_light={}
    for filename, interface in data.items():
        light_feature = interface['light_feature']
        heavy_feature=interface['heavy_feature']
        similarities_heavy[filename] = F.cosine_similarity(target_tensor, heavy_feature[0], dim=0).sum()
        similarities_light[filename] = F.cosine_similarity(target_tensor, light_feature[0], dim=0).sum()

    return similarities_heavy,similarities_light
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--processed_dir', type=str, default='./data/processed/antigen_computed')
    parser.add_argument('--reset', action='store_true', default=False)
    parser.add_argument('--model_config_path', type=str, required=True)
    parser.add_argument('--model_checkpoint', type=str, required=True)
    parser.add_argument('--tensor_directory', type=str, required=True)
    args = parser.parse_args()
    feature_all=load_tensors_from_directory(args.tensor_directory)

    if args.reset:
        sure = input('Sure to reset? (y/n): ')
        if sure != 'y':
            exit()

    # 初始化模型
    config, config_name = load_config(args.model_config_path)
    print("building model...")
    model = ContrastiveDiffAb(config.model).to('cuda')
    print('building complete,starting load checkpoints...')
    # 加载 checkpoint
    checkpoint = torch.load(args.model_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('checkpoint loaded,start processing...')

    # 初始化数据集
    dataset = SAbDabDatasetWithtopn(
        model=model,
        processed_dir=args.processed_dir,
        split=args.split,
        reset=args.reset,
        feature_all=feature_all
    )

    # 打印样本和数据集信息
    print(dataset[114])
    print(len(dataset), len(dataset.clusters))
#python -m diffab.datasets.sabdab_antigen_computed --model_config_path ./configs/contrastive/embedding.yml --model_checkpoint ./logs/embedding_2024_10_07__01_17_28/checkpoints/epoch_200.pt --tensor_directory ./feature_data/compute_2024_10_07__16_28_21_compute/checkpoints --reset