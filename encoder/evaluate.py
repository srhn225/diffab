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
    parser = argparse.ArgumentParser(description="Amino Acid Encoding Evaluation")

    # General parameters
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint to load')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for computation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')

    args = parser.parse_args()
    return args

def evaluate(args, model, loader, logger, device='cuda'):
    """
    模型评估函数

    参数：
    - args: 命令行参数
    - model: ContrastiveDiffAb模型
    - loader: 数据加载器
    - logger: 日志记录器
    - device: 计算设备（cuda或cpu）
    """
    
    # 评估模式
    logger.info('Starting evaluation...')
    model.eval()  # 将模型设置为评估模式
    total_loss = 0  # 总损失
    num_batches = len(loader)

    with torch.no_grad():  # 禁用梯度计算（评估阶段不需要）
        with tqdm(total=len(loader), desc="Evaluating") as pbar:
            for batch_idx, batch in enumerate(loader):
                # 递归将 batch 中所有张量移动到指定设备上（如GPU）
                batch = recursive_to(batch, device)

                # 前向传播并计算损失
                loss = model(batch)
                total_loss += loss.item()

                pbar.set_postfix(loss=" {:.8f}".format(loss.item()))  # 在进度条中显示当前损失
                pbar.update(1)

    # 计算平均损失
    avg_loss = total_loss / num_batches
    logger.info(f"Evaluation complete. Average loss: {avg_loss:.8f}")

if __name__ == "__main__":
    args = parse_args()  # 解析命令行参数

    # 加载配置文件
    config, config_name = load_config(args.config)

    # 设置日志和结果保存路径
    log_dir = get_new_log_dir(args.save_dir, prefix=config_name, tag='eval')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

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
        collate_fn=PaddingCollate_unmerged(),  # 根据需要自定义的 collate 函数
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

    logger.info(f"Checkpoint loaded, starting evaluation...")

    # 调用 evaluate 函数进行评估
    evaluate(
        args=args,             # 命令行参数
        model=model,           # ContrastiveDiffAb 模型
        loader=loader,         # 数据加载器
        logger=logger,         # 日志记录器
        device=args.device      # 计算设备（cuda 或 cpu）
    )
