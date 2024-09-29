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
    parser = argparse.ArgumentParser(description="Amino Acid Encoding")

    # General parameters
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models and logs')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for computation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')

    # Logging and checkpointing
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to store TensorBoard logs')
    parser.add_argument('--save_freq', type=int, default=10, help='Frequency (in epochs) to save model checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to a checkpoint to resume from')

    args = parser.parse_args()
    return args

def train(args, model, loader, optimizer, scheduler, logger, writer, device='cuda'):
    """
    训练模型的主函数

    参数：
    - args: 命令行参数
    - model: ContrastiveDiffAb模型
    - loader: 数据加载器
    - optimizer: 优化器
    - scheduler: 学习率调度器
    - logger: 日志记录器
    - writer: TensorBoard记录器
    - device: 计算设备（cuda或cpu）
    """

    # 开始训练
    logger.info('Starting training...')
    model.train()  # 将模型设置为训练模式
    global_step = 0  # 用于记录全局步数
    for epoch in range(config.train.max_iters):
        logger.info(f"Epoch {epoch+1}/{config.train.max_iters}")

        # 遍历整个数据集
        with tqdm(total=len(loader), desc=f"Epoch {epoch+1}") as pbar:
            for batch_idx, batch in enumerate(loader):
                # 递归将 batch 中所有张量移动到指定设备上（如GPU）
                batch = recursive_to(batch, device)
                # stats = torch.cuda.memory_summary()
                # print(stats)

                # 前向传播
                loss = model(batch)
                loss.backward()  # 反向传播计算梯度
                # 反向传播和优化步骤
                optimizer.zero_grad()  # 梯度清零
                
                optimizer.step()  # 优化器更新参数

                # 更新学习率调度器（如果有）
                # if scheduler is not None:
                #     scheduler.step()

                # 记录训练损失到 TensorBoard
                writer.add_scalar('train/loss', loss.item(), global_step)
                pbar.set_postfix(loss=loss.item())  # 在进度条中显示当前损失
                pbar.update(1)

                global_step += 1
                del loss, batch

        # 保存模型检查点
        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(args.save_dir, f'epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, save_path)
            logger.info(f"Model saved to {save_path}")

        # 每个epoch结束时记录学习率（如果使用了调度器）
        # if scheduler is not None:
        #     writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)

        torch.cuda.empty_cache()
    logger.info('Training complete.')

if __name__ == "__main__":
    args = parse_args()  # 解析命令行参数

    # 加载配置文件
    config, config_name = load_config(args.config)

    # 设置日志和检查点保存路径
    if args.resume:
        log_dir = os.path.dirname(os.path.dirname(args.resume))
    else:
        log_dir = get_new_log_dir(args.log_dir, prefix=config_name, tag='')
    
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(ckpt_dir): 
        os.makedirs(ckpt_dir)
    
    # 日志记录器
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)

    # 复制配置文件到日志目录
    if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
        shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    logger.info(args)
    logger.info(config)

    # 数据加载器
    logger.info('Loading dataset...')
    dataset = get_dataset(config.dataset)
    loader = DataLoader(
        dataset, 
        batch_size=config.train.batch_size, 
        collate_fn=PaddingCollate_unmerged(),  # 根据需要自定义的 collate 函数
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    logger.info(f'Dataset loaded, total {len(dataset)} samples.')

    # 模型初始化
    model = ContrastiveDiffAb(config.model).to(args.device)



    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()

    # 从检查点恢复训练（如果提供了恢复路径）
    if args.resume:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict']) 
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info(f'Resumed training from epoch {start_epoch}')
    else:
        start_epoch = 0

    # 调用 train 函数开始训练
    train(
        args=args,              # 命令行参数
        model=model,            # ContrastiveDiffAb 模型
        loader=loader,          # 数据加载器
        optimizer=optimizer,    # 优化器
        scheduler=scheduler,    # 学习率调度器（可选）
        logger=logger,          # 日志记录器
        writer=writer,          # TensorBoard 记录器
        device=args.device      # 计算设备（cuda 或 cpu）
    )

