import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence_channelwise(p, q):
    b, c, h, w = p.shape
    p_flat = p.view(b, c, -1).clamp(-50, 50)
    q_flat = q.view(b, c, -1).clamp(-50, 50)
    p_dist = F.softmax(p_flat, dim=-1).clamp(min=1e-8)
    q_dist = F.softmax(q_flat, dim=-1).clamp(min=1e-8)
    kl = (p_dist * (p_dist.log() - q_dist.log())).sum(dim=-1)
    return kl


class ImportanceGuidedAggregation(nn.Module):
    def __init__(self):
        super(ImportanceGuidedAggregation, self).__init__()

    def forward(self, student_feat, guidance_pool):
        if len(guidance_pool) == 0:
            return torch.zeros_like(student_feat)

        if len(guidance_pool) == 1:
            return guidance_pool[0]

        dist_list = []
        for g in guidance_pool:
            d = kl_divergence_channelwise(student_feat, g)
            dist_list.append(d)

        dist_stack = torch.stack(dist_list, dim=-1)

        importance = F.softmax(dist_stack, dim=-1)  # (B, C, num_teachers)

        aggregated = torch.zeros_like(student_feat)
        for idx, g in enumerate(guidance_pool):
            w = importance[:, :, idx].unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            aggregated = aggregated + w * g

        return aggregated


class TeacherGuidance:
    def __init__(self):
        self.dam_guidance = []
        self.restore_guidance = []

    def update(self, dam_out, restore_out):
        self.dam_guidance.append(dam_out.detach())
        self.restore_guidance.append(restore_out.detach())

    @property
    def group1(self):
        return self.dam_guidance

    @property
    def group2(self):
        return self.restore_guidance

    def clear(self):
        self.dam_guidance = []
        self.restore_guidance = []

    def __len__(self):
        return len(self.dam_guidance)
