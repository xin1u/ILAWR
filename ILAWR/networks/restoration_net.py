import torch
import torch.nn as nn
import torch.nn.functional as F


class _LayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        return weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (gx,
                (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
                grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
                None)


class ChannelLayerNorm(nn.Module):
    def __init__(self, n_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_channels))
        self.bias = nn.Parameter(torch.zeros(n_channels))
        self.eps = eps

    def forward(self, x):
        return _LayerNormFn.apply(x, self.weight, self.bias, self.eps)


class GatedUnit(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dw_expand=2, ffn_expand=2,
                 dropout_rate=0.):
        super().__init__()
        dw_ch = channels * dw_expand

        self.spatial_conv1 = nn.Conv2d(channels, dw_ch, 1)
        self.spatial_dw = nn.Conv2d(
            dw_ch, dw_ch, kernel_size,
            padding=(kernel_size - 1) // 2, groups=dw_ch)
        self.spatial_conv2 = nn.Conv2d(dw_ch // 2, channels, 1)

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch // 2, dw_ch // 2, 1),
        )
        self.gate = GatedUnit()

        ffn_ch = ffn_expand * channels
        self.ffn_conv1 = nn.Conv2d(channels, ffn_ch, 1)
        self.ffn_conv2 = nn.Conv2d(ffn_ch // 2, channels, 1)

        self.norm1 = ChannelLayerNorm(channels)
        self.norm2 = ChannelLayerNorm(channels)

        self.drop1 = nn.Dropout(dropout_rate) if dropout_rate > 0. else nn.Identity()
        self.drop2 = nn.Dropout(dropout_rate) if dropout_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.spatial_conv1(x)
        x = self.spatial_dw(x)
        x = self.gate(x)
        x = x * self.channel_attn(x)
        x = self.spatial_conv2(x)
        x = self.drop1(x)
        y = inp + x * self.beta

        x = self.ffn_conv1(self.norm2(y))
        x = self.gate(x)
        x = self.ffn_conv2(x)
        x = self.drop2(x)
        return y + x * self.gamma


class WeatherRestorationNet(nn.Module):
    def __init__(self, in_channels=3, width=32, middle_blk_num=1,
                 enc_blk_nums=None, dec_blk_nums=None, kernel_size=3):
        super().__init__()
        if enc_blk_nums is None:
            enc_blk_nums = [1, 1, 1, 28]
        if dec_blk_nums is None:
            dec_blk_nums = [1, 1, 1, 1]

        self.head = nn.Conv2d(in_channels, width, 3, padding=1)
        self.tail = nn.Conv2d(width, 3, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        ch = width
        for n_blks in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(ch, kernel_size) for _ in range(n_blks)]))
            self.downsamples.append(nn.Conv2d(ch, ch * 2, 2, 2))
            ch *= 2

        self.bottleneck = nn.Sequential(
            *[NAFBlock(ch, kernel_size) for _ in range(middle_blk_num)])

        for n_blks in dec_blk_nums:
            self.upsamples.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 1, bias=False),
                nn.PixelShuffle(2)))
            ch //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(ch, kernel_size) for _ in range(n_blks)]))

        self.padder_size = 2 ** len(enc_blk_nums)
        self.feat_dim = width

    def forward(self, inp, return_feats=False):
        B, C, H, W = inp.shape
        x = self._pad(inp)
        base = x[:, :3, :, :]

        x = self.head(x)

        skips = []
        for enc, down in zip(self.encoders, self.downsamples):
            x = enc(x)
            skips.append(x)
            x = down(x)

        deep_feat = self.bottleneck(x)
        x = deep_feat

        for dec, up, skip in zip(self.decoders, self.upsamples, skips[::-1]):
            x = up(x)
            x = x + skip
            x = dec(x)

        out = self.tail(x)
        out = out + base
        out = out[:, :, :H, :W]

        if return_feats:
            return out, deep_feat
        return out

    def _pad(self, x):
        _, _, h, w = x.size()
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_w, 0, pad_h))

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


if __name__ == "__main__":
    net = WeatherRestorationNet(
        width=32, middle_blk_num=6,
        enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
    x = torch.randn(1, 3, 256, 256)
    y = net(x)
    print('output:', y.shape)
    print('#params:', sum(p.numel() for p in net.parameters()) / 1e6, 'M')

    y2, feat = net(x, return_feats=True)
    print('deep feat:', feat.shape)
