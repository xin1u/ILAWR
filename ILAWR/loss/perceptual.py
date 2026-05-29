import torch
import torch.nn as nn
from torchvision import models


class VGG19Features(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19Features, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.block1 = nn.Sequential()
        self.block2 = nn.Sequential()
        self.block3 = nn.Sequential()
        self.block4 = nn.Sequential()
        self.block5 = nn.Sequential()

        for i in range(2):
            self.block1.add_module(str(i), vgg[i])
        for i in range(2, 7):
            self.block2.add_module(str(i), vgg[i])
        for i in range(7, 12):
            self.block3.add_module(str(i), vgg[i])
        for i in range(12, 21):
            self.block4.add_module(str(i), vgg[i])
        for i in range(21, 30):
            self.block5.add_module(str(i), vgg[i])

        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        h1 = self.block1(x)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        h4 = self.block4(h3)
        h5 = self.block5(h4)
        return [h1, h2, h3, h4, h5]


class ContrastivePerceptualLoss(nn.Module):
    def __init__(self, device, tau=0.5):
        super(ContrastivePerceptualLoss, self).__init__()
        self.vgg = VGG19Features().to(device)
        self.l1 = nn.L1Loss()
        self.layer_weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.tau = tau

    def forward(self, restored, clean, degraded):
        feat_restored = self.vgg(restored)
        feat_clean = self.vgg(clean)
        feat_degraded = self.vgg(degraded)
        loss = []
        for i in range(len(feat_restored)):
            sim_pos = torch.exp(-self.l1(feat_restored[i], feat_clean[i]) / self.tau)
            sim_neg = torch.exp(-self.l1(feat_restored[i], feat_degraded[i]) / self.tau)
            nce = -torch.log(sim_pos / (sim_pos + sim_neg))
            loss.append(self.layer_weights[i] * nce)
        return sum(loss)
