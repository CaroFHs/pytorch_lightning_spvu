import torch
from collections import OrderedDict
from torch.autograd import Variable
import time

from torch import nn
from pytorch_lightning.core import LightningModule

from vugan.vugan_select import get_activ_layer
import yaml

## Generator
from vugan.unet00 import UNet # UNet
from vugan.transformer00 import PixelwiseViT # ViT

## Discriminator
from vugan.unet_discriminator import Unet_Discriminator
from vugan.PixelDiscriiminator import PixelDiscriminator
from vugan.sed import SeD_P, SeD_U

# 读取 YAML 配置文件
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 获取模型配置
model_config = config['model']
training_config = config['training']
data_config = config['data']
G_config = config['Generator']


class TransUNet_G(LightningModule):
    def __init__(self, 

        ## ViT
        features = G_config['input_layer_output']*8, 
        n_heads = 6, 
        n_blocks = 6, 
        ffn_features = G_config['input_layer_output']*8*2,
        embed_features = G_config['input_layer_output']*8,
        vit_activ = G_config['vit_activ'], 
        vit_norm = G_config['vit_norm'], 
        use_ffn = G_config['vit_use_ffn'], # vit是否使用ffn
        rezero          = True,

        ## UNet
        image_shape = [3, 512, 512], 
        input_nc = 3,  # 输入层输入通道数
        output_nc = 1, # 输出层输出通道数
        input_layer_out = G_config['input_layer_output'], # 改，添加输入层输出通道数参数
        unet_features_list = [48, 96, 192, 384],  # unet每层输出特征图通道数

        unet_activ = G_config['unet_activ'], 
        unet_norm = G_config['unet_norm'],
        unet_downsample = G_config['unet_downsample'], # unet下采样方法
        unet_upsample   = 'upsample-conv', # unet上采样方法

        decoder_use_FHAB = G_config['decoder_use_FHAB'], # decoder是否使用FHAB
        encoder_use_FFDF = G_config['encoder_use_FFDF'], # decoder是否使用FFDF
        unet_use_dropout = G_config['unet_use_dropout'], # unet是否使用dropout
        unet_rezero     = False,
        activ_output    = None  

        ):
        super(TransUNet_G, self).__init__()



        self.image_shape = image_shape # 存储图像形状
        self.output_nc   = output_nc # 改，增加输出通道参数，不与输入通道共用参数image_shape[0]

        self.net = UNet(
            unet_features_list, unet_activ, unet_norm, image_shape, 
            input_layer_out, # 改，添加输入层输出通道数
            decoder_use_FHAB, # 改，添加decoder中选择使用FHAB
            encoder_use_FFDF, # 改，添加decoder中选择使用FFDF
            unet_use_dropout, # 改，添加unet_vit中选择使用dropout
            input_nc, # 改，添加输入层的输入通道数input_nc
            output_nc, unet_downsample, unet_upsample, unet_rezero
        ) # 初始化UNet模型

        bottleneck = PixelwiseViT(
            features, n_heads, n_blocks, ffn_features, embed_features,
            vit_activ, vit_norm,
            image_shape = self.net.get_inner_shape(),
            use_ffn = use_ffn, # 改，添加选择使用ffn的opt参数
            rezero      = rezero
        ) # 初始化PixelwiseViT 模型，作为 U-Net 的瓶颈层

        self.net.set_bottleneck(bottleneck) # PixelwiseViT 作为瓶颈层插入到 U-Net 模型中

        self.output = get_activ_layer(activ_output) # 初始化输出层的激活函数

    def forward(self, x):
        # x : (N, C, H, W)
        result = self.net(x)

        return self.output(result)


class DiscriminatorManager(LightningModule):
    def __init__(self, which_model_netD,
        input_nc = model_config['input_nc'],   # 输入层输入通道数
        output_nc = model_config['output_nc'], # 输出层输入通道数
        ndf = G_config['input_layer_output'],  # 第一层卷积输入通道数
        use_dropout = G_config['unet_use_dropout'], # unet是否使用dropout
        resolution = model_config['output_size'],  # 输入分辨率
        ):
        super(DiscriminatorManager, self).__init__()

        self.which_model_netD = which_model_netD

        if which_model_netD == 'unetdiscriminator':
            self.netD = Unet_Discriminator(
                input_c = input_nc,
                resolution = resolution,
                )
        elif which_model_netD == 'pixel':
            self.netD = PixelDiscriminator(
                input_nc = input_nc,
                gpu_ids = [0]
            )
        elif which_model_netD == 'sed_p':
            self.netD = SeD_P(
                input_nc= input_nc,
                # ndf = G_config['input_layer_output'],
            )
        elif which_model_netD == 'sed_u':
            self.netD = SeD_U(
                num_in_ch = input_nc, # input_layer channel number
                num_feat = 64, # unet_input channel number
            )
        else:
            raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                    which_model_netD)
        
    def forward(self, x, semantic):
        if self.which_model_netD in {'sed_p', 'sed_u'}:
            return self.netD(x, semantic)
        else:
            return self.netD(x)
    
