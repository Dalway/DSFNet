import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from models.model_util import *
from train_util import *
from pytorch_wavelets import DWTForward, DWTInverse

# define the function includes in import *
__all__ = [
    'SpixelNet1l','SpixelNet1l_bn'
]
#没有小波变换

class SpixelNet(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(SpixelNet,self).__init__()

        self.batchNorm = batchNorm
        self.assign_ch = 9

        self.stem = Stem()



        self.conv1a = conv(self.batchNorm, 16, 32, kernel_size=3, stride=2)
        self.conv2a = conv(self.batchNorm, 32, 64, kernel_size=3, stride=2)
        self.conv3a = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv4a = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)


        self.op1 = Partial_decouple(32,4)
        self.op2 = Partial_decouple(64,4)
        self.op3 = Partial_decouple(128,4)
        self.op4 = Partial_decouple(256,4)



##########+128+64+32
        self.channel_mapping = nn.Sequential(
            nn.Conv2d(256, 9 * 32, kernel_size=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2d(9 * 32, 9 * 32, kernel_size=3, padding=1, dilation=1, bias=True, groups=9 * 32)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.sde_module = SDE_module(9 * 32, 9 * 32)

        self.channel_backing = nn.Sequential(
            nn.Conv2d(9 * 32, 256, kernel_size=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1, bias=True, groups=256)
        )
##########
        self.block1 = ctmBlock(dim=32, mlp_ratio=4,  drop=0., drop_path=0.)
        self.block2 = ctmBlock(dim=64, mlp_ratio=4,  drop=0., drop_path=0.)
        self.block3 = ctmBlock(dim=128, mlp_ratio=4,  drop=0., drop_path=0.)
        self.block4 = ctmBlock(dim=256, mlp_ratio=4,  drop=0., drop_path=0.)

        # self.uppp = nn.Sequential(
        #
        #     nn.PixelShuffle(4),
        #     conv(self.batchNorm, 16, 16, kernel_size=3),
        #
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     conv(self.batchNorm, 16, 16, kernel_size=3),
        #
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     conv(self.batchNorm, 16, 16, kernel_size=3),
        #
        # )
        #self.uppp = AuxiliaryHead(256)
##########
        self.combine1 = SpaceBlock(16,16,16)
        self.combine2 = SpaceBlock(32,32,32)
        self.combine3 = SpaceBlock(64,64,64)
        self.combine4 = SpaceBlock(128,128,128)
##########
        self.deconv3 = deconv(256, 128)
        self.conv3_1 = conv(self.batchNorm, 256, 128)
        self.pred_mask3 = predict_mask(128, self.assign_ch)

        self.deconv2 = deconv(128, 64)
        self.conv2_1 = conv(self.batchNorm, 128, 64)
        self.pred_mask2 = predict_mask(64, self.assign_ch)

        self.deconv1 = deconv(64, 32)
        self.conv1_1 = conv(self.batchNorm, 64, 32)
        self.pred_mask1 = predict_mask(16, self.assign_ch)

        self.deconv0 = deconv(32, 16)
        self.conv0_1 = conv(self.batchNorm, 32, 16)
        self.pred_mask0 = predict_mask(16,self.assign_ch)

        self.softmax = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)



    def forward(self, x):
        #out1 = self.conv0b(self.conv0a(x))  # 5*5

        c1 = self.stem(x)
###################
        down1 = self.conv1a(c1)
        #test = down1.clone()
        out1 = self.op1(down1)  # 32, 104, 104

        # are_equal = torch.equal(test, down1)
        #
        # print(are_equal)  # 输出: True
        skip1 = self.block1(out1) #+ down1
#####################
        down2 =self.conv2a(skip1)
        out2 = self.op2(down2)

        skip2 = self.block2(out2) #+ down2
#####################
        down3 = self.conv3a(skip2)
        out3 = self.op3(down3)  # 47*47

        skip3 = self.block3(out3) #+ down3
######################
        down4 = self.conv4a(skip3)
        out4 = self.op4(down4)  # 95*95

        skip4 = self.block4(out4) #+ down4
#####################
        da = self.channel_mapping(skip4)
        d_prior = self.gap(da)
        decouple = self.sde_module(da, d_prior)
        decouple = self.channel_backing(decouple)

#####################
        # auxout = self.uppp(decouple)
        # auxout = self.pred_mask1(auxout)
        # auxout = self.softmax(auxout)

        # are_equal = torch.equal(test, decouple)
        #
        # print(are_equal)  # 输出: True
###################

        out_deconv3 = self.deconv3(decouple)
        aux4 = self.combine4(out_deconv3, skip3)
        concat3 = torch.cat((aux4, out_deconv3), 1)
        out_conv3_1 = self.conv3_1(concat3)

        out_deconv2 = self.deconv2(out_conv3_1)
        aux3 = self.combine3(out_deconv2, skip2)
        concat2 = torch.cat((aux3, out_deconv2), 1)
        out_conv2_1 = self.conv2_1(concat2)

        out_deconv1 = self.deconv1(out_conv2_1)
        aux2 = self.combine2(out_deconv1, skip1)
        concat1 = torch.cat((aux2, out_deconv1), 1)
        out_conv1_1 = self.conv1_1(concat1)

        out_deconv0 = self.deconv0(out_conv1_1)
        aux1 = self.combine1(out_deconv0, c1)
        concat0 = torch.cat((aux1, out_deconv0), 1)
        out_conv0_1 = self.conv0_1(concat0)

        mask0 = self.pred_mask0(out_conv0_1)
        prob0 = self.softmax(mask0)

        return prob0#, auxout

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

class AuxiliaryHead(nn.Module):
    def __init__(self, in_c):
        super(AuxiliaryHead, self).__init__()
        self.branch_fg = nn.Sequential(
            conv(True, in_c, 256, kernel_size=3, stride=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            conv(True, 256, 128, kernel_size=3, stride=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            conv(True, 128, 64, kernel_size=3, stride=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            conv(True, 64, 32, kernel_size=3, stride=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            conv(True, 32, 16, kernel_size=3, stride=1),
            #conv(True, 16, 16, kernel_size=1, stride=1),
        )


    def forward(self, x):

        x = self.branch_fg(x)
        return x

class Stem(nn.Module):

    def __init__(self, stem_width=16):
        super(Stem, self).__init__()
        self.conv1 =nn.Sequential(
            nn.Conv2d(3, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.LeakyReLU(0.1),
            nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.LeakyReLU(0.1),

        )

    def forward(self, x):
        out = self.conv1(x)
        return out


class WaveTransform(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.up_wave = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.DWT = DWTForward(J=1, wave='haar')#.cuda()

        self.attn = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 4, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding=1, groups=dim)
        )
        self.v1 = nn.Conv2d(dim, dim, 1)
        self.attn1 = nn.Conv2d(dim * 4, dim, 1)
        self.proj1 = nn.Conv2d(dim, dim, 1)

        self.v2 = nn.Conv2d(dim * 3, dim * 3, 1)
        self.attn2 = nn.Conv2d(dim * 4, dim * 3, 1)
        self.proj2 = nn.Conv2d(dim * 3, dim * 3, 1)

        #self.rir = RB(dim * 4)

        self.IDWT = DWTInverse(wave='haar')#.cuda()
        #self.down_wave = nn.Upsample(scale_factor=1./2, mode='bilinear', align_corners=True)

        self.wave_output = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 4, 1,padding=0),
            nn.Conv2d(dim * 4, dim * 4, 3, 1, 1, bias=True, groups=dim * 4),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim * 4, 1,padding=0)
        )
        self.wave_output1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        a = DMT1_yh[0]
        list_tensor.append(DMT1_yl)
        for i in range(3):
            list_tensor.append(a[:, :, i, :, :])
        return torch.cat(list_tensor, 1)

    def _Itransformer(self, out):
        yh = []
        C = int(out.shape[1] / 4)
        # print(out.shape[0])
        y = out.reshape((out.shape[0], C, 4, out.shape[-2], out.shape[-1]))
        yl = y[:, :, 0].contiguous()
        yh.append(y[:, :, 1:].contiguous())

        return yl, yh


    def forward(self, x: Tensor) -> Tensor:

        x = self.up_wave(x)

        DMT0_yl, DMT0_yh = self.DWT(x)
        DMT0 = self._transformer(DMT0_yl, DMT0_yh)

        xl, xh = torch.split(DMT0, [self.dim, self.dim*3], dim=1)
        attn = self.attn(DMT0)

        xl = self.attn1(attn) * self.v1(xl)
        xl = self.proj1(xl)

        xh = self.attn2(attn) * self.v2(xh)
        xh = self.proj2(xh)

        x = torch.cat((xl,xh),1)
        x = self.wave_output(x)
        DMT0 = self._Itransformer(x)
        IDMT = self.IDWT(DMT0)

        x = self.wave_output1(IDMT)
        return x

#选择删去了特征选择门，频域和空间域解耦
class Partial_decouple(nn.Module):

    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        #self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1)

        #LKD Visual Attention Network
        self.cheap_operation1 = Attention(self.dim_conv3, 0)
        self.cheap_operation2 = Attention(self.dim_conv3, 1)
        self.wave_operation2 = WaveTransform(self.dim_conv3 * 2)
        self.gate = Interactive_Fusion(self.dim_conv3 * 2)

        self.identity = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

#spatial and frequency devided to half of two

    def forward(self, x: Tensor) -> Tensor:
        # for training/inference
        spatial, frequency = torch.split(x, [self.dim_conv3 * 2, self.dim_conv3 * 2], dim=1)
        spatial1, spatial2 = torch.split(spatial, [self.dim_conv3, self.dim_conv3], dim=1)

        spatial1 = self.cheap_operation1(spatial1)
        spatial2 = self.cheap_operation2(spatial2)
        spatial = torch.cat((spatial1, spatial2), dim=1)
        frequency = self.wave_operation2(frequency)

        gate_x = self.gate(spatial,frequency)

        return gate_x + self.identity(x)

class  Interactive_Fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_s = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),#是否需要进行groups
            nn.Sigmoid()
        )
        self.gate_f = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),#是否需要进行groups
            nn.Sigmoid()
        )

    def forward(self, s,f):
        gate_s = self.gate_s(s)
        gate_f = self.gate_f(f)

        s_g = gate_s * s + (1-gate_s) * f + s
        f_g = gate_f * f + (1-gate_f) * s + f

        fusion = torch.cat((s_g,f_g),1)
        return fusion


class LKA(nn.Module):
    def __init__(self, dim,k):
        super().__init__()
        self.k = k
        if self.k == 1:
            self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
            self.conv1 = nn.Conv2d(dim, dim, 1)
        else:
            self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)
            self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Module):
    def __init__(self, d_model,k):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model,k)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class ctmMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )

        self.proj = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2d(hidden_features, eps=1e-5)

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
class ctmBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=2., drop=0., drop_path=0.):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ctmMlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):

        x = x + self.drop_path(self.mlp(x))

        return x


class SDE_module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SDE_module, self).__init__()
        self.inter_channels = in_channels // 9

        self.att1 = DANetHead(self.inter_channels, self.inter_channels)
        self.att2 = DANetHead(self.inter_channels, self.inter_channels)
        self.att3 = DANetHead(self.inter_channels, self.inter_channels)
        self.att4 = DANetHead(self.inter_channels, self.inter_channels)
        self.att5 = DANetHead(self.inter_channels, self.inter_channels)
        self.att6 = DANetHead(self.inter_channels, self.inter_channels)
        self.att7 = DANetHead(self.inter_channels, self.inter_channels)
        self.att8 = DANetHead(self.inter_channels, self.inter_channels)
        self.att9 = DANetHead(self.inter_channels, self.inter_channels)

        self.final_conv = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels, out_channels, 1))
        # self.encoder_block = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels, 32, 1))

        self.reencoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        )

    def forward(self, x, d_prior):

        ### re-order encoded_c5 ###

        enc_feat = self.reencoder(d_prior)

        feat1 = self.att1(x[:, :self.inter_channels], enc_feat[:, 0:self.inter_channels])
        feat2 = self.att2(x[:, self.inter_channels:2 * self.inter_channels],
                          enc_feat[:, self.inter_channels:2 * self.inter_channels])
        feat3 = self.att3(x[:, 2 * self.inter_channels:3 * self.inter_channels],
                          enc_feat[:, 2 * self.inter_channels:3 * self.inter_channels])
        feat4 = self.att4(x[:, 3 * self.inter_channels:4 * self.inter_channels],
                          enc_feat[:, 3 * self.inter_channels:4 * self.inter_channels])
        feat5 = self.att5(x[:, 4 * self.inter_channels:5 * self.inter_channels],
                          enc_feat[:, 4 * self.inter_channels:5 * self.inter_channels])
        feat6 = self.att6(x[:, 5 * self.inter_channels:6 * self.inter_channels],
                          enc_feat[:, 5 * self.inter_channels:6 * self.inter_channels])
        feat7 = self.att7(x[:, 6 * self.inter_channels:7 * self.inter_channels],
                          enc_feat[:, 6 * self.inter_channels:7 * self.inter_channels])
        feat8 = self.att8(x[:, 7 * self.inter_channels:8 * self.inter_channels],
                          enc_feat[:, 7 * self.inter_channels:8 * self.inter_channels])
        feat9 = self.att8(x[:, 8 * self.inter_channels:9 * self.inter_channels],
                          enc_feat[:, 8 * self.inter_channels:9 * self.inter_channels])
        feat = torch.cat([feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8,feat9], dim=1)

        sasc_output = self.final_conv(feat)
        sasc_output = sasc_output + x

        return sasc_output


class DANetHead(nn.Module):
    def __init__(self, in_channels, inter_channels, norm_layer=nn.BatchNorm2d):
        super(DANetHead, self).__init__()
        # inter_channels = in_channels // 8
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, padding=0, bias=False),#kernelsize变了。。。。。。。。。。。
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, padding=0, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1,groups=inter_channels, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1,groups=inter_channels, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, inter_channels, 1))

    def forward(self, x, enc_feat):
        feat1 = self.conv5a(x)
        sa_feat = feat1
        sa_conv = self.conv51(sa_feat)
        # sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = feat2
        sc_conv = self.conv52(sc_feat)
        # sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        feat_sum = feat_sum * torch.sigmoid(enc_feat)

        sasc_output = self.conv8(feat_sum)

        return sasc_output

class SpaceBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 channel_in,
                 out_channels,
                 scale_aware_proj=False):
        super(SpaceBlock, self).__init__()
        self.scale_aware_proj = scale_aware_proj

        # self.scene_encoder = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(out_channels, out_channels, 1),
        # )
        #变了这个前景编码
        self.scene_encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
        )
        self.content_encoders = nn.Sequential(
            nn.Conv2d(channel_in, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.feature_reencoders = nn.Sequential(
            nn.Conv2d(channel_in, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature, features):
        content_feats = self.content_encoders(features)
        scene_feature = F.adaptive_avg_pool2d(scene_feature, (1, 1))

        scene_feat = self.scene_encoder(scene_feature)
        relations = self.normalizer((scene_feat * content_feats).sum(dim=1, keepdim=True))

        p_feats = self.feature_reencoders(features)

        refined_feats = relations * p_feats

        return refined_feats


def SpixelNet1l( data=None):
    # Model without  batch normalization
    model = SpixelNet(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def SpixelNet1l_bn(data=None):
    # model with batch normalization
    model = SpixelNet(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
#
if __name__ == '__main__':
    model = SpixelNet1l_bn().cuda()
    model.train()
    a = torch.rand(1, 3, 208, 208).cuda()
    b,c = model(a)
    print(b.shape, '\t',c.shape)
    # model = SDE_module(81,81,81).cuda(),c.shape
    # a= torch.rand(1,81,13,13).cuda()
    # b = torch.rand(1,81,1,1).cuda()
    # c = model(a,b)
    # print(c.shape)