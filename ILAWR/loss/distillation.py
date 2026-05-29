import torch
import torch.nn as nn
import torch.nn.functional as F


class DAMDistillationLoss(nn.Module):
    def __init__(self):
        super(DAMDistillationLoss, self).__init__()

    def forward(self, student_dam_out, teacher_guidance):
        b, c, h, w = student_dam_out.shape
        s_flat = student_dam_out.view(b, c, -1)
        t_flat = teacher_guidance.view(b, c, -1)

        s_flat = s_flat.clamp(-50, 50)
        t_flat = t_flat.clamp(-50, 50)

        s_log = F.log_softmax(s_flat, dim=-1)
        t_prob = F.softmax(t_flat, dim=-1)

        kl = F.kl_div(s_log, t_prob, reduction='batchmean')
        return kl


class ReconstructionDistillationLoss(nn.Module):
    def __init__(self, contrast_loss_fn=None, alpha_contrast=0.3):
        super(ReconstructionDistillationLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.contrast_fn = contrast_loss_fn
        self.alpha_contrast = alpha_contrast

    def forward(self, student_out, teacher_guidance, degraded_input=None):
        loss = self.l1(student_out, teacher_guidance)
        if self.contrast_fn is not None and degraded_input is not None:
            loss = loss + self.alpha_contrast * self.contrast_fn(
                student_out, teacher_guidance, degraded_input)
        return loss
