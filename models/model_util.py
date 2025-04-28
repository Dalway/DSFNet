import torch
import torch.nn as nn

## *************************** my functions ****************************

def predict_param(in_planes, channel=3):
    return  nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True)

def predict_mask(in_planes, channel=9):
    return  nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True)

def predict_feat(in_planes, channel=20, stride=1):
    return  nn.Conv2d(in_planes, channel, kernel_size=3, stride=stride, padding=1, bias=True)

def predict_prob(in_planes, channel=9):
    return  nn.Sequential(
        nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True),
        nn.Softmax(1)
    )
#***********************************************************************

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1)
        )


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1)
    )

import torch.nn.functional as F
from torch import Tensor
# MRLA is medium-range lightweight Attention
class MRLA(nn.Module):
    def __init__(self, channel, att_kernel):
        super(MRLA, self).__init__()
        att_padding = att_kernel // 2
        self.gate_fn = nn.Sigmoid()
        self.channel = channel
        channels12 = int(channel / 2)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(channel, channels12, 1, 1, bias=False),
            nn.BatchNorm2d(channels12),
            nn.GELU(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(channels12, channels12, 3, 1, 1, groups=channels12, bias=False),
            nn.BatchNorm2d(channels12),
            nn.GELU(),
        )
        self.init = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, bias=False),
            nn.BatchNorm2d(channel),
        )
        self.H_att = nn.Conv2d(channel, channel, (att_kernel, 1), 1, (att_padding, 0), groups=channel, bias=False)
        self.V_att = nn.Conv2d(channel, channel, (1, att_kernel), 1, (0, att_padding), groups=channel, bias=False)
        self.batchnorm = nn.BatchNorm2d(channel)

    def forward(self, x):
        x_tem = self.init(F.avg_pool2d(x, kernel_size=2, stride=2))
        x_h = self.H_att(x_tem)
        x_w = self.V_att(x_tem)
        mrla = self.batchnorm(x_h + x_w)

        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = out[:, :self.channel, :, :] * F.interpolate(self.gate_fn(mrla),
                                                          size=(out.shape[-2], out.shape[-1]),
                                                          mode='nearest')
        return out


# GA is long range Attention
class GA(nn.Module):
    def __init__(self, dim, head_dim=4, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 3, 1, 2)
        return x


# MBFD is multi-branch feature decoupling module
class MBFD(nn.Module):

    def __init__(self, dim, stage, att_kernel):
        super().__init__()
        self.dim = dim
        self.stage = stage
        self.dim_learn = dim // 4
        self.dim_untouched = dim - self.dim_learn - self.dim_learn
        self.Conv = nn.Conv2d(self.dim_learn, self.dim_learn, 3, 1, 1, bias=False)
        self.MRLA = MRLA(self.dim_learn, att_kernel)  # MRLA is medium range Attention
        if stage > 2:
            self.GA = GA(self.dim_untouched)      # GA is long range Attention
            self.norm = nn.BatchNorm2d(self.dim_untouched)

    def forward(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2, x3 = torch.split(x, [self.dim_learn, self.dim_learn, self.dim_untouched], dim=1)
        x1 = self.Conv(x1)
        x2 = self.MRLA(x2)
        if self.stage > 2:
            x3 = self.norm(x3 + self.GA(x3))
        x = torch.cat((x1, x2, x3), 1)

        return x
