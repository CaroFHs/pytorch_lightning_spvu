# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

import torch
from torch import nn

from vugan.vugan_select import get_norm_layer, get_activ_layer

from .cnn import get_downsample_x2_layer, get_upsample_x2_layer
from vugan.modules.deconv import DEC

import matplotlib.pyplot as plt

class UnetBasicBlock(nn.Module):
    ''' 构建U-Net架构中的基本卷积块 '''

    def __init__(
        self, in_features, out_features, activ, norm, mid_features = None,
        **kwargs
    ):
        '''
        in_features：   输入特征通道数
        out_features：  输出特征通道数
        activ：         激活函数
        norm：          归一化类型
        mid_features：  中间特征图的通道数

        '''
        super().__init__(**kwargs)

        if mid_features is None: # 若中间特征未指定
            mid_features = out_features # 则设置为等于out_features

        ## 定义基本块：归一化层 + 卷积层 + 归一化层 + 激活函数
        self.block = nn.Sequential(
            get_norm_layer(norm, in_features),
            nn.Conv2d(in_features, mid_features, kernel_size = 3, padding = 1),
            get_activ_layer(activ),

            get_norm_layer(norm, mid_features),
            nn.Conv2d(
                mid_features, out_features, kernel_size = 3, padding = 1
            ),
            get_activ_layer(activ),
        )

    def forward(self, x):
        return self.block(x) # 返回构建好的UNet基本块

class UNetEncBlock(nn.Module):
    ''' 构建UNet的encoder块，主要负责特征的下采样和特征图的初步提取 '''

    def __init__(
        self, features, activ, norm, downsample, input_shape, **kwargs
    ):
        '''
        features：      本层输出特征图通道数
        activ：         激活函数
        norm：          归一化类型
        downsample：    下采样方法
        input_shape：   输入特征图形状 (C, H, W)
        '''
        super().__init__(**kwargs)

        ## 调用get_downsample_x2_layer函数获取下采样层，该层会将特征图的空间尺寸减半，并可能改变通道数。
        # output_features记录下采样后特征图的通道数。
        self.downsample, output_features = \
            get_downsample_x2_layer(downsample, features)

        (C, H, W)  = input_shape
        self.block = UnetBasicBlock(C, features, activ, norm) # 创建UNet基本块

        ################################ 改，encoder后加DEConv ###############################
        self.DEConv = DEC(features)

        #####################################################################################


        self.output_shape = (output_features, H//2, W//2) # 采样后的输出特征图的形状（空间尺寸减半）

    def get_output_shape(self):
        ''' 获取编码器块输出特征图的形状 '''
        return self.output_shape

    def forward(self, x):

        ## 通过UnetBasicBlock进行特征提取,得到的结果，用于跳跃连接，即传递给相应的解码器块
        r = self.block(x)
        ##### 改，加DEConv #####
        r = self.DEConv(r)
        #######################


        ## 通过下采样层self.downsample，得到的特征图,用于传递给下一个编码器块或解码器块
        y = self.downsample(r) 
        ##### 改，加DEConv #####
        y = self.DEConv(y)
        #######################

        return (y, r)

class UNetDecBlock(nn.Module):

    def __init__(
        self, output_shape, activ, norm, upsample, input_shape,
        rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.upsample, input_features = get_upsample_x2_layer(
            upsample, input_shape[0]
        )

        self.block = UnetBasicBlock(
            2 * input_features, output_shape[0], activ, norm,
            mid_features = max(input_features, input_shape[0])
        )

        ################################ 改，backbone输入decoder前加DEConv ###############################
        self.DEConv = DEC(dim=input_shape[0])

        #####################################################################################

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self, x, r):
        # x : (N, C, H_in, W_in)
        # r : (N, C, H_out, W_out)

        # x : (N, C_up, H_out, W_out)
        ##### 改，加DEConv #####
        x = self.DEConv(x)
        #######################
        x = self.re_alpha * self.upsample(x)

        # y : (N, C + C_up, H_out, W_out)
        y = torch.cat([x, r], dim = 1)

        # result : (N, C_out, H_out, W_out)
        return self.block(y)

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )

class UNetBlock(nn.Module):
    ''' 构建UNetBlock '''
    def __init__(
        self, features, activ, norm, image_shape, downsample, upsample,
        rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.conv = UNetEncBlock(
            features, activ, norm, downsample, image_shape
        )

        self.inner_shape  = self.conv.get_output_shape()
        self.inner_module = None

        self.deconv = UNetDecBlock(
            image_shape, activ, norm, upsample, self.inner_shape, rezero
        )

    def get_inner_shape(self):
        return self.inner_shape

    def set_inner_module(self, module):
        self.inner_module = module

    def get_inner_module(self):
        return self.inner_module

    def forward(self, x):
        # x : (N, C, H, W)

        # y : (N, C_inner, H_inner, W_inner)
        # r : (N, C_inner, H, W)
        (y, r) = self.conv(x)

        # y : (N, C_inner, H_inner, W_inner)
        y = self.inner_module(y)

        # y : (N, C, H, W)
        y = self.deconv(y, r)

        return y

class UNet(nn.Module):

    def __init__(
        self, features_list, activ, norm, image_shape, output_nc, downsample, upsample,
        rezero = True, **kwargs
    ):
        '''
        features_list:  每个U-Net块的特征数量列表。
        activ:          激活函数。
        norm:           归一化方法。
        image_shape:    输入图像的形状。
        downsample 和 upsample: 下采样和上采样方法。
        rezero:         是否使用ReZero技术。
        '''
        # pylint: disable = too-many-locals
        super().__init__(**kwargs)

        self.features_list = features_list
        self.image_shape   = image_shape

        self.output_nc     = output_nc # 改，增加输出通道参数，不与输入通道共用参数image_shape[0]

        self._construct_input_layer(activ) # 定义输入层
        self._construct_output_layer(output_nc)     # 定义输出层

        unet_layers = []
        curr_image_shape = (features_list[0], *image_shape[1:])

        ## 循环遍历features_list，为每一层创建一个UNetBlock对象，UNetBlock负责处理
        # 下采样、编码、解码和上采样的过程，并更新当前特征图的形状。
        for features in features_list:
            layer = UNetBlock(
                features, activ, norm, curr_image_shape, downsample, upsample,
                rezero
            )
            curr_image_shape = layer.get_inner_shape()
            unet_layers.append(layer)

        ## 连接U-Net的编码器和解码器部分，即设置每一层的内部模块，
        # 使上一层能访问下一层的输出，用于跳跃连接。
        for idx in range(len(unet_layers)-1):
            unet_layers[idx].set_inner_module(unet_layers[idx+1])

        self.unet = unet_layers[0] # 设置self.unet为U-Net的第一层，即整个U-Net网络的入口

    def _construct_input_layer(self, activ):
        ''' 定义输入层：一个卷积层 + 一个激活层 '''
        self.layer_input = nn.Sequential(
            nn.Conv2d(
                self.image_shape[0], self.features_list[0],
                kernel_size = 3, padding = 1
            ), # 卷积层将输入通道数转换为第一个特征层的通道数
            get_activ_layer(activ),
        )

    def _construct_output_layer(self, output_nc):
        ''' 定义输出层：一个1x1卷积层。将特征通道数转换回输入图像的通道数 '''
        self.layer_output = nn.Conv2d(
            self.features_list[0], output_nc, kernel_size = 1
        ) # 改，将输出维度改为opt.output_nc，原为self.image_shape[0]

    def get_innermost_block(self):
        ''' 获取U-Net结构中的最内层块 '''
        result = self.unet

        for _ in range(len(self.features_list)-1):
            result = result.get_inner_module()

        return result

    def set_bottleneck(self, module):
        ''' 设置瓶颈层，可直接输入bottleneck模块 '''
        self.get_innermost_block().set_inner_module(module)

    def get_bottleneck(self):
        ''' 返回瓶颈层 '''
        return self.get_innermost_block().get_inner_module()

    def get_inner_shape(self):
        ''' 获取最内层块的形状 '''
        return self.get_innermost_block().get_inner_shape()

    def forward(self, x):
        # x : (N, C, H, W) = [1, 3, 256, 256]

        y = self.layer_input(x)  # [1, 48, 256, 256]
        y_i = self.unet(y)         # [1, 48, 256, 256]
        y_o = self.layer_output(y_i) # [1, 3, 256, 256]

        # img = x.permute((0, 2, 3, 1))
        # # img = img.cpu().numpy()
        # img = img[0]
        # plt.subplot(221)
        # plt.imshow(img)
        # plt.title("original")

        # plt.subplot(222)
        # plt.imshow(y_o)
        # plt.title("output")

        # # plt.subplot(223)
        # # plt.imshow(y_i,cmap="rgb")
        # # plt.title("centralization")

        # # # plt.subplot(224)
        # # # plt.imshow(F_log,cmap="rgb")
        # # # plt.title("F_log")

        # plt.suptitle('square image FFT') # 设置标题
        # plt.tight_layout() 
        # plt.show()


        return y_o

