# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

import torch
from torch import nn

from vugan.vugan_select import get_norm_layer, get_activ_layer

from .cnn import get_downsample_x2_layer, get_upsample_x2_layer

import matplotlib.pyplot as plt

from vugan.modules.hab import FeatureHybridAttentionBlock
from vugan.modules.FFDF import Frequency_Feature_Dynamic_Fusion
# from models.spvu_Encoder_with_DE.vugan.modules.MEEM import DE
from vugan.modules.GMP2d_DE import DE


class UnetBasicBlock(nn.Module):
    ''' 构建U-Net架构中的基本卷积块 '''

    def __init__(
        self, in_features, out_features, unet_activ, unet_norm, 
        use_dropout, # 改，添加unet_vit中选择使用dropout
        mid_features = None, **kwargs
    ):
        '''
        in_features：   输入特征通道数
        out_features：  输出特征通道数
        unet_activ：         激活函数
        unet_norm：          归一化类型
        mid_features：  中间特征图的通道数

        '''
        super().__init__(**kwargs)

        if mid_features is None: # 若中间特征未指定
            mid_features = out_features # 则设置为等于out_features

        ## 定义基本块：归一化层 + 卷积层 + 归一化层 + 激活函数

        ######################## 改，更改basicblock定义方法，便于添加dropout #######################
        layers = [
            get_norm_layer(unet_norm, in_features),
            nn.Conv2d(in_features, mid_features, kernel_size=3, padding=1),
            get_activ_layer(unet_activ),
        ]

        if use_dropout:
            layers.append(nn.Dropout2d(p=0.2))  # 添加Dropout层

        layers.extend([
            get_norm_layer(unet_norm, mid_features),
            nn.Conv2d(mid_features, out_features, kernel_size=3, padding=1),
            get_activ_layer(unet_activ),
        ])

        if use_dropout:
            layers.append(nn.Dropout2d(p=0.2))  # 再次添加Dropout层

        self.block = nn.Sequential(*layers)
        ########################################################################################

        ##################################### 上方改动原代码 ######################################
        # self.block = nn.Sequential(
        #     get_norm_layer(norm, in_features),
        #     nn.Conv2d(in_features, mid_features, kernel_size = 3, padding = 1),
        #     get_activ_layer(activ),

        #     get_norm_layer(norm, mid_features),
        #     nn.Conv2d(
        #         mid_features, out_features, kernel_size = 3, padding = 1
        #     ),
        #     get_activ_layer(activ),
        # )
        ########################################################################################

    def forward(self, x):
        return self.block(x) # 返回构建好的UNet基本块

class UNetEncBlock(nn.Module):
    ''' 构建UNet的encoder块，主要负责特征的下采样和特征图的初步提取 '''

    def __init__(
        self, features, unet_activ, unet_norm, downsample, input_shape, 
        encoder_use_FFDF, # 改，添加decoder中选择使用FFDF
        use_dropout, # 改，添加unet_vit中选择使用dropout
        **kwargs
    ):
        '''
        features：      本层输出特征图通道数
        unet_activ：         激活函数
        unet_norm：          归一化类型
        downsample：    下采样方法
        input_shape：   输入特征图形状 (C, H, W)
        '''
        super().__init__(**kwargs)

        self.downsample = downsample # 改，添加downsample类型名称，用于选择forward过程
        ## 调用get_downsample_x2_layer函数获取下采样层，该层会将特征图的空间尺寸减半，并可能改变通道数。
        # output_features记录下采样后特征图的通道数。
        self.downsample_layer, output_features = \
            get_downsample_x2_layer(downsample, features)

        (C, H, W)  = input_shape
        self.block = UnetBasicBlock(C, features, unet_activ, unet_norm,
                                    use_dropout, # 改，添加unet_vit中选择使用dropout
                                    ) # 创建UNet基本块

        self.output_shape = (output_features, H//2, W//2) # 采样后的输出特征图的形状（空间尺寸减半）

        ################################################################################
        if encoder_use_FFDF == True:
            self.ffdf = Frequency_Feature_Dynamic_Fusion(nc=features)
        ################################################################################

    def get_output_shape(self):
        ''' 获取编码器块输出特征图的形状 '''
        return self.output_shape

    def forward(self, x):
        if self.downsample in ['wtfd_cat', 'wtfd_add']:
            x = self.block(x) # 特征图x输入basicblock，通道数*2
            x_H, x_L = self.downsample_layer(x) # 再输入wtfd下采样模块，得到高、低频增强特征图
            r = x_H # 高频用于跳跃连接，传递给相应的解码器块
            y = x_L # 低频用于传递给下一个编码器块
        else:
            ## 通过UnetBasicBlock进行特征提取,得到的结果，用于跳跃连接，即传递给相应的解码器块
            r = self.block(x)
            ## 通过下采样层self.downsample，得到的特征图,用于传递给下一个编码器块
            y = self.downsample_layer(r)

        return (y, r)

class UNetDecBlock(nn.Module):

    def __init__(
        self, output_shape, unet_activ, unet_norm, upsample, input_shape,
        decoder_use_FHAB, # 改，添加decoder中选择使用FHAB
        use_dropout, # 改，添加unet_vit中选择使用dropout
        rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.upsample, input_features = get_upsample_x2_layer(
            upsample, input_shape[0]
        ) # get_upsample_x2_layer处理结果就是input_features = input_shape[0]；通道数不变，尺寸×2

        self.block = UnetBasicBlock(
            2 * input_features, output_shape[0], unet_activ, unet_norm,
            use_dropout, # 改，添加unet_vit中选择使用dropout
            mid_features = max(input_features, input_shape[0])
        ) # basicblock，通道数减半，尺寸不变

    ########################## 改，添加特征混合注意力模块fhab ##########################
        self.decoder_use_FHAB = decoder_use_FHAB
        if self.decoder_use_FHAB == True:
            self.fhab = FeatureHybridAttentionBlock(
                input_dim=output_shape[0], input_shape=output_shape[1], patch_size=4)
            # basicblock的输出作为FHAB的输入
        else:
            pass
    ################################################################################

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self, x, r):
        # x : (N, C, H_in, W_in)
        # r : (N, C, H_out, W_out)

        # x : (N, C_up, H_out, W_out)
        x = self.re_alpha * self.upsample(x) # 下层decoder输出的x上采样，再乘rezero系数

        # y : (N, C + C_up, H_out, W_out)
        y = torch.cat([x, r], dim = 1) # skip输入和上采样后的x concat

        # y : (N, C_out, H_out, W_out)
        y = self.block(y) # 经过basicblock

    ########################## 改，添加特征混合注意力模块fhab ##########################
        if self.decoder_use_FHAB == True:
            # y : (N, C_out, H_out, W_out)
            y = self.fhab(y) # 经过FHAB
        else:
            pass
    ################################################################################

        # result : (N, C_out, H_out, W_out)
        return y

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )

class UNetBlock(nn.Module):
    ''' 构建UNetBlock '''
    def __init__(
        self, features, unet_activ, unet_norm, image_shape, downsample, upsample,
        decoder_use_FHAB, # 改，添加decoder中选择使用FHAB
        encoder_use_FFDF, # 改，添加decoder中选择使用FFDF
        use_dropout, # 改，添加unet_vit中选择使用dropout
        rezero = True, **kwargs
    ):
        super().__init__(**kwargs)

        self.conv = UNetEncBlock(
            features, unet_activ, unet_norm, downsample, image_shape,
            encoder_use_FFDF, # 改，添加decoder中选择使用FHAB
            use_dropout, # 改，添加unet_vit中选择使用dropout
        )

        self.inner_shape  = self.conv.get_output_shape()
        self.inner_module = None

        self.deconv = UNetDecBlock(
            image_shape, unet_activ, unet_norm, upsample, self.inner_shape, 
            decoder_use_FHAB, # 改，添加decoder中选择使用FHAB
            use_dropout, # 改，添加unet_vit中选择使用dropout
            rezero
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
        self, features_list, unet_activ, unet_norm, image_shape, 
        input_layer_out, # 改，添加输入层输出通道数
        decoder_use_FHAB, # 改，添加decoder中选择使用FHAB
        encoder_use_FFDF, # 改，添加decoder中选择使用FFDF
        use_dropout, # 改，添加unet_vit中选择使用dropout
        input_nc, # 改，添加输入层的输入通道数input_nc
        output_nc, downsample, upsample, rezero = True, **kwargs
    ):
        '''
        features_list:  每个U-Net块的特征数量列表。
        unet_activ:          激活函数。
        unet_norm:           归一化方法。
        image_shape:    输入图像的形状。
        downsample 和 upsample: 下采样和上采样方法。
        rezero:         是否使用ReZero技术。
        '''
        # pylint: disable = too-many-locals
        super().__init__(**kwargs)

        self.features_list = features_list
        self.image_shape   = image_shape
        self.input_layer_out = input_layer_out # 改，添加输入层输出通道数

        self.input_nc = input_nc, # 改，添加输入层的输入通道数input_nc
        self.output_nc = output_nc # 改，增加输出通道参数，不与输入通道共用参数image_shape[0]

        self._construct_input_layer(unet_activ,input_nc,input_layer_out) # 定义输入层
        self._construct_output_layer(input_layer_out,output_nc)     # 定义输出层

        unet_layers = []
        curr_image_shape = (self.input_layer_out, *image_shape[1:]) # 改，第一个unetblock输入：48→24

        ## 循环遍历features_list，为每一层创建一个UNetBlock对象，UNetBlock负责处理
        # 下采样、编码、解码和上采样的过程，并更新当前特征图的形状。
        for features in features_list:
            layer = UNetBlock(
                features, unet_activ, unet_norm, curr_image_shape, downsample, upsample,
                decoder_use_FHAB, # 改，添加decoder中选择使用FHAB
                encoder_use_FFDF, # 改，添加decoder中选择使用FFDF
                use_dropout, # 改，添加unet_vit中选择使用dropout
                rezero
            )
            curr_image_shape = layer.get_inner_shape()
            unet_layers.append(layer)

        ## 连接U-Net的编码器和解码器部分，即设置每一层的内部模块，
        # 使上一层能访问下一层的输出，用于跳跃连接。
        for idx in range(len(unet_layers)-1):
            unet_layers[idx].set_inner_module(unet_layers[idx+1])

        self.unet = unet_layers[0] # 设置self.unet为U-Net的第一层，即整个U-Net网络的入口


#########################################################################################
    def _construct_input_layer(self, unet_activ, input_nc, input_layer_out):
        # ''' 输入层使用细节增强模块，(b,3,h,w) → (b,48,h,w) '''
        # self.layer_input = DE(c_in=input_nc, img_dim=input_layer_out, 
        #              norm=nn.BatchNorm2d, act=nn.LeakyReLU) # DE使用的norm和unet不同

        ''' 定义输入层：一个卷积层 + 一个激活层 '''
        self.layer_input = nn.Sequential(
            nn.Conv2d(
                self.image_shape[0], self.input_layer_out, # 改，更改input层输出，48→24
                kernel_size = 3, padding = 1
            ), # 卷积层将输入通道数转换为第一个特征层的输入通道数
            get_activ_layer(unet_activ),
        )

    def _construct_output_layer(self, input_layer_out, output_nc):
        ''' 输出层接收 unet的输出 和 输入层的输出，(b,48+48,h,w) → (b,1,h,w) '''
        self.layer_output = nn.Sequential(
            nn.Conv2d(input_layer_out*2, input_layer_out, kernel_size = 1),
            nn.Conv2d(input_layer_out, output_nc, kernel_size = 1),
            nn.Tanh()  # 改，输出层添加 tanh 激活函数将输出限制在 [-1, 1]
        )

        # ''' 定义输出层：一个1x1卷积层。将特征通道数转换回输入图像的通道数 '''
        # self.layer_output = nn.Conv2d(
        #     self.input_layer_out, output_nc, kernel_size = 1 # 改，更改output层输出，48→24
        # ) # 改，将输出维度改为opt.output_nc，原为self.image_shape[0]
#########################################################################################


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
        y_input = self.layer_input(x)  # [1, 48, 256, 256]
        y_unet = self.unet(y_input)         # [1, 48, 256, 256]
        y_de = torch.cat([y_input, y_unet], dim = 1) # [1, 48+48, 256, 256]
        y_o = self.layer_output(y_de) # [1, 1, 256, 256]

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

