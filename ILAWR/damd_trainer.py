import os
import time
import torch
import numpy as np
from copy import deepcopy
from torch import optim

from networks.aggregation import ImportanceGuidedAggregation
from loss.distillation import DAMDistillationLoss, ReconstructionDistillationLoss
from utils.metrics import compute_psnr, compute_ssim
from utils.helpers import ensure_dir
from utils.scheduler import cosine_decay_lr


class DAMDTrainer:
    def __init__(self, net, dam, criterion, train_loaders, test_loaders,
                 device, logger, args):
        self.net = net
        self.dam = dam
        self.criterion = criterion
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.logger = logger
        self.device = device
        self.args = args

        self.igam = ImportanceGuidedAggregation()
        self.kl_loss = DAMDistillationLoss()
        self.recon_distill_loss = ReconstructionDistillationLoss(
            contrast_loss_fn=criterion[1] if len(criterion) > 1 else None,
            alpha_contrast=args.alpha2
        )

        self.dam_teachers = []
        self.net_teachers = []

    def _infinite_loader(self, loader):
        """Yield batches endlessly by cycling through the DataLoader."""
        while True:
            for batch in loader:
                yield batch

    def run_session(self, session_id):
        """Train one incremental session."""
        self.net.unfreeze()
        self.dam.unfreeze()
        optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, self.net.parameters())) +
            list(filter(lambda p: p.requires_grad, self.dam.parameters())),
            lr=self.args.lr, betas=(0.9, 0.999), eps=1e-8)

        total_steps = self.args.steps
        best_ssim = 0
        best_psnr = 0
        t0 = time.time()

        task_name = self.args.task_order[session_id]
        ckpt_dir = os.path.join(self.args.save_dir, self.args.exp_name, task_name)
        ensure_dir(ckpt_dir)

        data_iter = self._infinite_loader(self.train_loaders[session_id])

        for step in range(1, total_steps + 1):
            self.net.train()
            self.dam.train()

            if not self.args.no_lr_sche:
                lr = cosine_decay_lr(step, total_steps, self.args.lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            else:
                lr = self.args.lr

            x, y = next(data_iter)
            x, y = x.to(self.device), y.to(self.device)

            optimizer.zero_grad()

            loss, loss_base, loss_t1, loss_t2 = self._compute_loss(
                x, y, session_id)
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.net.parameters()) + list(self.dam.parameters()),
                max_norm=1.0)
            optimizer.step()

            elapsed = (time.time() - t0) / 60
            print(f'\rbase:{loss_base.item():.5f} |kl:{loss_t1.item():.5f} '
                  f'|recon:{loss_t2.item():.5f} |step:{step}/{total_steps} '
                  f'|lr:{lr:.7f} |time:{elapsed:.1f}min', end='', flush=True)

            if step % self.args.eval_step == 0:
                torch.save(self.net.state_dict(),
                           os.path.join(ckpt_dir, f'net_step{step}.pth'))
                avg_ssim, avg_psnr = self.evaluate(session_id)
                msg = f'step:{step} |ssim:{avg_ssim:.4f} |psnr:{avg_psnr:.2f}'
                print(f'\n{msg}')
                self.logger.write(msg + '\n')

                if avg_ssim >= best_ssim and avg_psnr >= best_psnr:
                    best_ssim = avg_ssim
                    best_psnr = avg_psnr
                    torch.save({
                        'step': step,
                        'best_ssim': best_ssim,
                        'best_psnr': best_psnr,
                        'net': self.net.state_dict(),
                        'dam': self.dam.state_dict(),
                    }, os.path.join(ckpt_dir, 'best_model.pk'))
                    save_msg = (f'saved at step:{step} '
                                f'|ssim:{best_ssim:.4f} |psnr:{best_psnr:.2f}')
                    print(save_msg)
                    self.logger.write(save_msg + '\n')

        best_path = os.path.join(ckpt_dir, 'best_model.pk')
        if not os.path.exists(best_path):
            torch.save({
                'step': total_steps,
                'best_ssim': 0,
                'best_psnr': 0,
                'net': self.net.state_dict(),
                'dam': self.dam.state_dict(),
            }, best_path)

    def _compute_loss(self, x, y, session_id):

        loss_t1 = torch.zeros(1, device=self.device)
        loss_t2 = torch.zeros(1, device=self.device)

        # Eq.1-3: base restoration loss on current data
        out, feats = self.net(x, return_feats=True)
        dam_out = self.dam(feats)

        loss_base = self.criterion[0](out, y)
        if self.args.use_contrast and len(self.criterion) > 1:
            loss_base = loss_base + self.args.alpha1 * self.criterion[1](out, y, x)

        total = loss_base

        if session_id > 0:
            # Eq.11,19: Teacher Group 1 — DAM distillation on current x
            guide1 = self.igam(dam_out, self._get_teacher_dam_outputs(x))
            loss_t1 = self.kl_loss(dam_out, guide1)

            # Eq.13,19: Teacher Group 2 — restoration distillation on current x
            guide2 = self.igam(out, self._get_teacher_restore_outputs(x))
            loss_t2 = self.recon_distill_loss(out, guide2, x)

            # Eq.20: total training loss
            total = total + self.args.lam * loss_t1 + self.args.zeta * loss_t2

        return total, loss_base, loss_t1, loss_t2

    def _get_teacher_dam_outputs(self, x):

        with torch.no_grad():
            for t_net, t_dam in zip(self.net_teachers, self.dam_teachers):
                _, f = t_net(x, return_feats=True)
                outputs.append(t_dam(f))
        return outputs

    def _get_teacher_restore_outputs(self, x):

        with torch.no_grad():
            for t_net in self.net_teachers:
                outputs.append(t_net(x))
        return outputs

    @torch.no_grad()
    def evaluate(self, session_id):
        self.net.eval()
        all_ssim = []
        all_psnr = []
        print()
        for loader in self.test_loaders[:session_id + 1]:
            ssims, psnrs = [], []
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                pred = self.net(inputs)
                ssims.append(compute_ssim(pred, targets).item())
                psnrs.append(compute_psnr(pred, targets))
            print(f'  psnr:{np.mean(psnrs):.4f} |ssim:{np.mean(ssims):.4f}')
            all_ssim.append(np.mean(ssims))
            all_psnr.append(np.mean(psnrs))
        return np.mean(all_ssim), np.mean(all_psnr)

    def after_session(self, session_id):
        task_name = self.args.task_order[session_id]
        ckpt_dir = os.path.join(self.args.save_dir, self.args.exp_name, task_name)
        best_ckpt = torch.load(
            os.path.join(ckpt_dir, 'best_model.pk'),
            map_location=self.device, weights_only=False)
        self.net.load_state_dict(best_ckpt['net'])
        self.dam.load_state_dict(best_ckpt['dam'])

        t_net = deepcopy(self.net)
        t_net.freeze()
        t_net.eval()
        self.net_teachers.append(t_net)

        t_dam = deepcopy(self.dam)
        t_dam.freeze()
        t_dam.eval()
        self.dam_teachers.append(t_dam)

        self.net.freeze()
        self.dam.freeze()
