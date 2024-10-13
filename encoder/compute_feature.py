import os
import shutil
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffab.datasets import get_dataset
from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.train import *
from encoder.embedding import diffabencoder
from encoder.models import ContrastiveDiffAb

def parse_args():
    parser = argparse.ArgumentParser(description="Computing features")

    # General parameters
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint to load')
    parser.add_argument('--save_dir', type=str, default='./feature_data', help='Directory to save data')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for computation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')

    # New parameter for saving raw data
    parser.add_argument('--save_raw', action='store_true', help='If set, will save raw heavy and light chain data')

    args = parser.parse_args()
    return args

def save_features(save_dir, data_id, heavy_feat, light_feat, antigen_feat, heavy_data=None, light_data=None):
    """
    保存计算出的特征为 .pt 文件

    参数：
    - save_dir: 保存文件的目录
    - data_id: 数据样本的唯一ID
    - heavy_feat: 计算出的 heavy chain 特征
    - light_feat: 计算出的 light chain 特征
    - antigen_feat: 计算出的 antigen 特征
    - heavy_data: 可选的 heavy chain 原始数据
    - light_data: 可选的 light chain 原始数据
    """
    os.makedirs(save_dir, exist_ok=True)
    feature_data = {
        'data_id': data_id,
        'heavy_feature': heavy_feat.cpu(),  # 转到CPU以便保存
        'light_feature': light_feat.cpu(),
        'antigen_feature': antigen_feat.cpu()
    }

    # 如果有原始数据需要保存
    if heavy_data is not None and light_data is not None:
        feature_data['heavy_data'] = heavy_data
        feature_data['light_data'] = light_data

    torch.save(feature_data, os.path.join(save_dir, f'{data_id}_features.pt'))

def compute(args, model, save_dir, loader, logger, device='cuda'):
    """
    计算并保存抗体和抗原的特征

    参数：
    - args: 命令行参数
    - model: ContrastiveDiffAb模型
    - loader: 数据加载器
    - logger: 日志记录器
    - device: 计算设备（cuda或cpu）
    """
    
    # 评估模式
    logger.info('Starting feature computation...')
    model.eval()  # 将模型设置为评估模式

    with torch.no_grad():  # 禁用梯度计算（评估阶段不需要）
        with tqdm(total=len(loader), desc="Computing Features") as pbar:
            for batch_idx, batch in enumerate(loader):
                # 递归将 batch 中所有张量移动到指定设备上（如GPU）
                batch = recursive_to(batch, device)

                # 获取抗原和抗体的ID
                data_id = batch['id']  # 确保每个batch中有ID

                # 计算抗体（heavy和light）的特征
                heavy_feat = model.compute_antibody_feature(batch['heavy'])
                light_feat = model.compute_antibody_feature(batch['light'])
                # 计算抗原的特征
                antigen_feat = model.compute_antigen_feature(batch['antigen'])

                # 检查是否需要保存原始 heavy 和 light 数据
                if args.save_raw:
                    heavy_data = batch['heavy']
                    light_data = batch['light']
                else:
                    heavy_data = None
                    light_data = None

                # 保存计算的特征和原始数据（如果有）
                save_features(save_dir, data_id, heavy_feat, light_feat, antigen_feat, heavy_data, light_data)

                pbar.update(1)

    logger.info("Feature computation complete.")

if __name__ == "__main__":
    args = parse_args()  # 解析命令行参数

    # 加载配置文件
    config, config_name = load_config(args.config)

    # 设置日志和结果保存路径
    log_dir = get_new_log_dir(args.save_dir, prefix=config_name, tag='compute')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(ckpt_dir): 
        os.makedirs(ckpt_dir)
    # 日志记录器
    logger = get_logger('eval', log_dir)

    logger.info(args)
    logger.info(config)

    # 数据加载器
    logger.info('Loading dataset...')
    dataset = get_dataset(config.dataset)
    loader = DataLoader(
        dataset, 
        batch_size=config.eval.batch_size, 
        collate_fn=PaddingCollate_unmerged_with_id(),  # 根据需要自定义的 collate 函数
        shuffle=False,  # 评估阶段不打乱数据
        num_workers=args.num_workers
    )
    logger.info(f'Dataset loaded, total {len(dataset)} samples.')

    # 模型初始化
    model = ContrastiveDiffAb(config.model).to(args.device)

    # 加载模型检查点
    logger.info(f'Loading checkpoint from {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    logger.info(f"Checkpoint loaded, starting feature computation...")

    # 调用 compute 函数计算特征并保存
    compute(
        args=args,             # 命令行参数
        model=model,           # ContrastiveDiffAb 模型
        save_dir=ckpt_dir,
        loader=loader,         # 数据加载器
        logger=logger,         # 日志记录器
        device=args.device      # 计算设备（cuda 或 cpu）
    )
#CUDA_VISIBLE_DEVICES=3 python -m encoder.compute_feature --config ./configs/contrastive/compute.yml --checkpoint ./logs/embedding_2024_10_07__01_17_28/checkpoints/epoch_200.pt