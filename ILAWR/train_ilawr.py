import os
import argparse
import torch
import torch.nn as nn

from damd_trainer import DAMDTrainer
from networks.restoration_net import WeatherRestorationNet
from networks.degradation_aware import DegradationAwareModule
from loss.perceptual import ContrastivePerceptualLoss
from datasets.weather_data import build_dataloaders
from utils.helpers import ensure_dir


parser = argparse.ArgumentParser(description='ILAWR Training')
parser.add_argument('--steps', type=int, default=500000)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--task_order', type=str, nargs='+', required=True)
parser.add_argument('--resume_tasks', type=int, nargs='+', required=False)
parser.add_argument('--eval_step', type=int, default=10000)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--alpha1', default=0.3, type=float,
                    help='contrast loss weight in base loss (Eq.3)')
parser.add_argument('--alpha2', default=0.2, type=float,
                    help='contrast weight in Teacher Group 2')
parser.add_argument('--lam', default=0.3, type=float,
                    help='lambda: weight for L_teach1 (Eq.20)')
parser.add_argument('--zeta', default=0.4, type=float,
                    help='zeta: weight for L_teach2 (Eq.20)')
parser.add_argument('--data_path', type=str, default='./datasets')
parser.add_argument('--log_dir', type=str, default='./logs')
parser.add_argument('--save_dir', type=str, default='./checkpoints/')

# NAFNet architecture
parser.add_argument('--base_width', type=int, default=32)
parser.add_argument('--middle_blk_num', type=int, default=1)
parser.add_argument('--enc_blks', nargs='+', type=int, default=[1, 1, 1, 28])
parser.add_argument('--dec_blks', nargs='+', type=int, default=[1, 1, 1, 1])
parser.add_argument('--kernel_size', type=int, default=3)

parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--use_contrast', type=bool, default=True)
parser.add_argument('--no_lr_sche', action='store_true', default=False)
parser.add_argument('--exp_name', type=str, default='ilawr_default')
args = parser.parse_args()


if __name__ == '__main__':
    ensure_dir(os.path.join(args.log_dir, args.exp_name))
    log_file = open(os.path.join(args.log_dir, args.exp_name, 'train.log'), 'a+')

    device = torch.device(args.device)

    net = WeatherRestorationNet(
        in_channels=3,
        width=args.base_width,
        middle_blk_num=args.middle_blk_num,
        enc_blk_nums=args.enc_blks,
        dec_blk_nums=args.dec_blks,
        kernel_size=args.kernel_size,
    ).to(device)

    # bottleneck channels = width * 2^(num_encoder_levels)
    bottleneck_ch = args.base_width * (2 ** len(args.enc_blks))
    dam = DegradationAwareModule(feat_channels=bottleneck_ch).to(device)

    print(f'#net params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M')
    print(f'#dam params: {sum(p.numel() for p in dam.parameters()) / 1e6:.2f}M')
    print(f'bottleneck channels: {bottleneck_ch}')

    train_loaders = build_dataloaders(args, split='train')
    test_loaders = build_dataloaders(args, split='test')

    criterion = [nn.L1Loss().to(device)]
    if args.use_contrast:
        criterion.append(ContrastivePerceptualLoss(device).to(device))

    trainer = DAMDTrainer(
        net=net, dam=dam, criterion=criterion,
        train_loaders=train_loaders, test_loaders=test_loaders,
        device=device, logger=log_file, args=args)

    for task_id in range(len(args.task_order)):
        task_name = args.task_order[task_id]
        print(f'\n{"="*60}')
        print(f'Session {task_id}: {task_name}')
        print(f'{"="*60}')
        log_file.write(f'session {task_id}: {task_name}\n')

        if args.resume_tasks is not None and task_id in args.resume_tasks:
            trainer.after_session(task_id)
        else:
            trainer.run_session(task_id)
            trainer.after_session(task_id)

    log_file.close()
