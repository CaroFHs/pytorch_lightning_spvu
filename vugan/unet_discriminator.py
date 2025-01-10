import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from . import layers


# class DiscriminatorManager(nn.Module):
#     def __init__(self, input_nc, output_nc=None, ngf=64, use_dropout=False, resolution=512):
#         super(DiscriminatorManager, self).__init__()



#         self.model = Unet_Discriminator(input_nc, resolution=resolution,
#                                         D_activation=nn.LeakyReLU(0.1, inplace=False), # nn.ReLU(inplace=False) or nn.LeakyReLU(0.1, inplace=False)
#                                         )

#         # self.model.init_weights()
#     def forward(self, inp):
#         return self.model(inp)


def D_unet_arch(input_c=3, ch=64, attention='64', ksize='333333', dilation='111111', out_channel_multiplier=1):
    arch = {}

    n = 2

    ocm = out_channel_multiplier

    # covers bigger perceptual fields
    arch[128] = {'in_channels': [input_c] + [ch * item for item in [1, 2, 4, 8, 16, 8 * n, 4 * 2, 2 * 2, 1 * 2, 1]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 16, 8, 4, 2, 1, 1]],
                 'downsample': [True] * 5 + [False] * 5,
                 'upsample': [False] * 5 + [True] * 5,
                 'resolution': [64, 32, 16, 8, 4, 8, 16, 32, 64, 128],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 11)}}

    arch[256] = {'in_channels': [input_c] + [ch * item for item in [1, 2, 4, 8, 8, 16, 8 * 2, 8 * 2, 4 * 2, 2 * 2, 1 * 2, 1]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 8, 8, 4, 2, 1, 1]],
                 'downsample': [True] * 6 + [False] * 6,
                 'upsample': [False] * 6 + [True] * 6,
                 'resolution': [128, 64, 32, 16, 8, 4, 8, 16, 32, 64, 128, 256],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 13)}}

    arch[512] = {'in_channels': [input_c] + [ch * item for item in [1, 2, 4, 8, 8, 16, 32, 16*2, 8 * 2, 8 * 2, 4 * 2, 2 * 2, 1 * 2, 1]],
                 'out_channels': [item * ch for item in [1, 2, 4, 8, 8, 16, 32, 16, 8, 8, 4, 2, 1, 1]],
                 'downsample': [True] * 7 + [False] * 7,
                 'upsample': [False] * 7 + [True] * 7,
                 'resolution': [256, 128, 64, 32, 16, 8, 4, 8, 16, 32, 64, 128, 256, 512],
                 'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')]
                               for i in range(2, 15)}}
    return arch


class Unet_Discriminator(nn.Module):

    def __init__(self, 
                 input_c, # 输入通道数 
                 D_ch=24, #unet第一层输入通道数
                 D_wide=True, # 
                 resolution=512, # 输入图像分辨率
                 D_kernel_size=3, # 卷积核大小，
                 D_attn='64', # 指定在什么分辨率的层应用自注意力机制
                 n_classes=1000, # 类别数量，如果使用条件 GAN，则需要指定
                 num_D_SVs=1, 
                 num_D_SV_itrs=1, 
                 D_activation=nn.LeakyReLU(0.1, inplace=False), # nn.ReLU(inplace=False) or nn.LeakyReLU(0.1, inplace=False),
                #  D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8, # adam优化器参数
                 SN_eps=1e-12, # 谱归一化的 epsilon 值
                #  output_dim=1, 
                #  D_mixed_precision=False, D_fp16=False,
                #  D_init='kaiming', # 权重初始化方法，默认采用 'ortho'（正交初始化）
                 skip_init=False, D_param='SN', decoder_skip_connection=True, **kwargs):
        super(Unet_Discriminator, self).__init__()

        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = D_activation
        # Initialization style
        # self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        # self.fp16 = D_fp16

        if self.resolution == 128:
            self.save_features = [0, 1, 2, 3, 4]
        elif self.resolution == 256:
            self.save_features = [0, 1, 2, 3, 4, 5]
        elif self.resolution == 512:
            self.save_features = [0, 1, 2, 3, 4, 5, 6]
        self.out_channel_multiplier = 1  # 4
        # Architecture
        self.arch = D_unet_arch(input_c, self.ch, self.attention, out_channel_multiplier=self.out_channel_multiplier)[resolution]

        self.unconditional = True  # kwargs["unconditional"]

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        if self.D_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear,
                                                  num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                  eps=self.SN_eps)

            self.which_embedding = functools.partial(layers.SNEmbedding,
                                                     num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                     eps=self.SN_eps)
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []

        for index in range(len(self.arch['out_channels'])):

            if self.arch["downsample"][index]:
                self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                                               out_channels=self.arch['out_channels'][index],
                                               which_conv=self.which_conv,
                                               wide=self.D_wide,
                                               activation=self.activation,
                                               preactivation=(index > 0),
                                               downsample=(
                                                   nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]

            elif self.arch["upsample"][index]:
                upsample_function = (
                    functools.partial(F.interpolate, scale_factor=2, mode="nearest")  # mode=nearest is default
                    if self.arch['upsample'][index] else None)

                self.blocks += [[layers.GBlock2(in_channels=self.arch['in_channels'][index],
                                                out_channels=self.arch['out_channels'][index],
                                                which_conv=self.which_conv,
                                                # which_bn=self.which_bn,
                                                activation=self.activation,
                                                upsample=upsample_function, skip_connection=True)]]

            # If attention on this block, attach it to the end
            attention_condition = index < 5
            if self.arch['attention'][self.arch['resolution'][index]] and attention_condition:  # index < 5
                print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
                print("index = ", index)
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                                     self.which_conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        #last_layer = nn.Conv2d(self.ch * self.out_channel_multiplier, 1, kernel_size=1)

        ############################# 改，去掉Sigmoid以适配wgangp ############################
        # last_layer = nn.Sequential(*[nn.Conv2d(self.ch * self.out_channel_multiplier, 1, kernel_size=1), nn.Sigmoid()]) # 下方改动原代码
        last_layer = nn.Conv2d(self.ch * self.out_channel_multiplier, 1, kernel_size=1)
        self.blocks.append(last_layer)
        ###################################################################################

        #
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        # self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)

        # self.linear_middle = self.which_linear(32 * self.ch if resolution == 512 else 16 * self.ch, output_dim) # 改，去掉最后的线性层
        # self.sigmoid = nn.Sigmoid()
        # Embedding for projection discrimination
        # if not kwargs["agnostic_unet"] and not kwargs["unconditional"]:
        #    self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1]+extra)


        # 如果不跳过初始化，则在所有成员变量定义完毕后进行初始化
        # if not skip_init:
        #     self.init_weights(init_type='kaiming')

    def init_weights(self, init_type='kaiming', gain=0.02):
        '''
        初始化模型的权重。
        :param init_type: 初始化类型 ('normal', 'xavier', 'kaiming', 'orthogonal')
        :param gain: 放大因子，用于调整初始化权重的标准差或尺度。
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


    def forward(self, x, y=None): # y为标签，用于条件gan
        # Stick x into h for cleaner for loops without flow control
        h = x

        residual_features = []
        residual_features.append(x)
        # Loop over blocks

        for index, blocklist in enumerate(self.blocks[:-1]):
            if self.resolution == 128:
                if index == 6:
                    h = torch.cat((h, residual_features[4]), dim=1)
                elif index == 7:
                    h = torch.cat((h, residual_features[3]), dim=1)
                elif index == 8:  #
                    h = torch.cat((h, residual_features[2]), dim=1)
                elif index == 9:  #
                    h = torch.cat((h, residual_features[1]), dim=1)

            elif self.resolution == 256:
                if index == 7:
                    h = torch.cat((h, residual_features[5]), dim=1)
                elif index == 8:
                    h = torch.cat((h, residual_features[4]), dim=1)
                elif index == 9:  #
                    h = torch.cat((h, residual_features[3]), dim=1)
                elif index == 10:  #
                    h = torch.cat((h, residual_features[2]), dim=1)
                elif index == 11:
                    h = torch.cat((h, residual_features[1]), dim=1)

            elif self.resolution == 512:
                if index == 8:
                    h = torch.cat((h, residual_features[6]), dim=1)
                elif index == 9:
                    h = torch.cat((h, residual_features[5]), dim=1)
                elif index == 10:  #
                    h = torch.cat((h, residual_features[4]), dim=1)
                elif index == 11:
                    h = torch.cat((h, residual_features[3]), dim=1)
                elif index == 12:  #
                    h = torch.cat((h, residual_features[2]), dim=1)
                elif index == 13:  #
                    h = torch.cat((h, residual_features[1]), dim=1)

            for block in blocklist:
                h = block(h)

            if index in self.save_features[:-1]:
                residual_features.append(h)

            if index == self.save_features[-1]:
                # Apply global sum pooling as in SN-GAN
                h_ = torch.sum(self.activation(h), [2, 3]) # 全局求和池化
                # Get initial class-unconditional output
                # bottleneck_out = self.linear_middle(h_)

                # bottleneck_out = self.sigmoid(bottleneck_out) # 改，去掉sigmoid
                
                # Get projection of final featureset onto class vectors and add to evidence

                # if self.unconditional:
                #     projection = 0
                # else:
                #     # this is the bottleneck classifier c
                #     emb_mid = self.embed_middle(y)
                #     projection = torch.sum(emb_mid * h_, 1, keepdim=True)
                # bottleneck_out = bottleneck_out + projection

        out = self.blocks[-1](h)
        out = torch.mean(out, [2, 3])  # 改，确保最终输出是标量

        # return out, bottleneck_out # 下方改动原代码
        return out # 改，只输出out，去掉bottleneck_out


"""
m = Unet_Discriminator()
x = torch.rand((23,3,128,128))
y = m(x)
print("ok")
"""

if __name__ == '_main__':
    d_model = Unet_Discriminator(input_c=1)
    print(d_model)
