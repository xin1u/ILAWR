import torch
import torch.nn as nn


class RDBS(nn.Module):
    def __init__(self, in_channels, n_dilations=None):
        super(RDBS, self).__init__()
        if n_dilations is None:
            n_dilations = [1, 3, 5]
        self.blocks = nn.ModuleList()
        for i, d in enumerate(n_dilations):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_channels + i * in_channels, in_channels, 3,
                          padding=d, dilation=d, bias=True),
                nn.GELU(),
            ))
        self.fusion = nn.Conv2d(in_channels * (len(n_dilations) + 1),
                                in_channels, 1, bias=True)

    def forward(self, x):
        features = [x]
        for block in self.blocks:
            out = block(torch.cat(features, dim=1))
            features.append(out)
        return x + self.fusion(torch.cat(features, dim=1))


class DegradationAwareModule(nn.Module):
    def __init__(self, feat_channels, weight_channels=None):
        super(DegradationAwareModule, self).__init__()
        if weight_channels is None:
            weight_channels = feat_channels
        self.avg_head = nn.Conv2d(1, 1, 3, padding=1, bias=True)
        self.fft_head = nn.Conv2d(feat_channels, feat_channels, 3, padding=1, bias=True)
        self.rdbs = RDBS(weight_channels, n_dilations=[1, 3, 5])
        self.out_conv = nn.Conv2d(weight_channels, weight_channels, 3, padding=1, bias=True)
        self.act = nn.GELU()

    def forward(self, feat):
        b, c, h, w = feat.shape


        g = feat.mean(dim=1, keepdim=True) 


        amp_avg = torch.fft.fft2(g).abs()         
        amp_avg = self.act(self.avg_head(amp_avg)) 
        amp_init = torch.fft.fft2(feat).abs()       
        amp_init = self.act(self.fft_head(amp_init)) 


        amp_specific = amp_init - amp_avg         


        w = self.act(self.rdbs(amp_specific))      

        return self.out_conv(w * feat)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


if __name__ == "__main__":
    dam = DegradationAwareModule(feat_channels=512)
    x = torch.randn(2, 512, 16, 16)
    out = dam(x)
    print(out.shape)
    print('#params:', sum(p.numel() for p in dam.parameters()) / 1e6, 'M')
