import torch
from torch import nn
from vugan.vugan_select import extract_name_kwargs

from pytorch_wavelets import DWTForward # 改，添加小波变换库


################################ 改，添加SE通道注意力模块 ################################
class SELayer_2d(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer_2d, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def forward(self, X_input):
        b, c, _, _ = X_input.size()  	# shape = [32, 64, 2000, 80]
        
        y = self.avg_pool(X_input)		# shape = [32, 64, 1, 1]
        y = y.view(b, c)				# shape = [32,64]
        
        # 第1个线性层（含激活函数），即公式中的W1，其维度是[channel, channer/16], 其中16是默认的
        y = self.linear1(y)				# shape = [32, 64] * [64, 4] = [32, 4]
        
        # 第2个线性层（含激活函数），即公式中的W2，其维度是[channel/16, channer], 其中16是默认的
        y = self.linear2(y) 			# shape = [32, 4] * [4, 64] = [32, 64]
        y = y.view(b, c, 1, 1)			# shape = [32, 64, 1, 1]， 这个就表示上面公式的s, 即每个通道的权重

        return X_input*y.expand_as(X_input)
################################################################################

########################### 改，添加Haar小波变换下采样方法 ###########################
class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_ch),   
                                    nn.ReLU(inplace=True),                                 
                                    ) 
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)        
        x = self.conv_bn_relu(x)

        return x
################################################################################

#################### 改，添加Haar小波变换频域增强方法（concat+卷积 合并版） ###################
class wtfd_cat(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(wtfd_cat, self).__init__()

        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        self.yH_downchannel_layer = nn.Sequential(
                            nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),
                            nn.BatchNorm2d(out_ch),   
                            nn.ReLU(inplace=True),                                 
                            ) # 原hwd卷积模块conv_bn_relu
        self.yL_downchannel_layer = nn.Sequential(
                            nn.Conv2d(in_ch*2, out_ch, kernel_size=1, stride=1),
                            nn.BatchNorm2d(out_ch),   
                            nn.ReLU(inplace=True),                                 
                            ) # 原hwd卷积模块conv_bn_relu
        self.yH_upsample_layer = nn.Sequential(
                        nn.Upsample(scale_factor = 2, mode='nearest'),
                        nn.Conv2d(in_ch*3, in_ch*3, kernel_size = 3, padding = 1),
                    ) # yH上采样，使用get_upsample_x2_upconv_layer的上采样方法
                    
        self.x_downsample_layer = nn.Conv2d(in_ch, out_ch, kernel_size = 2, stride = 2
                    ) # x下采样，使用get_downsample_x2_conv2_layer的下采样方法

    def forward(self, x):

        ## 提取高、低频信号
        yL, yH = self.wt(x) # yL是tensor，yH是list
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]

        ## 高频增强
        yH_cat = torch.cat([y_HL, y_LH, y_HH], dim=1) # 高频信号(n,2c*3,h/2,w/2)
        yH_cat = self.yH_upsample_layer(yH_cat) # 高频信号(n,2c*3,h/2,w/2)→(n,2c*3,h,w)
        x_H = torch.cat([x, yH_cat], dim=1) # 高频增强x(n,2c*4,h,w)
        x_H = self.yH_downchannel_layer(x_H) # 高频增强x(n,2c*4,h,w)→(n,2c,h,w)

        ## 低频增强
        x = self.x_downsample_layer(x)  # 原特征图x下采样(n,2c,h/2,w/2)
        x_L = torch.cat([x, yL], dim=1) # 低频增强x(n,2c*2,h/2,w/2)
        x_L = self.yL_downchannel_layer(x_L) # 低频增强x(n,2c*2,h/2,w/2)→(n,2c,h/2,w/2)

        return x_H, x_L

################################################################################
#################### 改，添加Haar小波变换频域增强方法（add+归一化 合并版） ###################
class wtfd_add(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(wtfd_add, self).__init__()

        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        self.x_InstanceNorm_layer = nn.BatchNorm2d(out_ch) # 高频实例归一化模块
        self.xds_InstanceNorm_layer = nn.BatchNorm2d(out_ch) # 高频实例归一化模块
        self.yH_InstanceNorm_layer = nn.BatchNorm2d(out_ch) # 高频实例归一化模块
        self.yL_InstanceNorm_layer = nn.BatchNorm2d(out_ch) # 低频实例归一化模块

                    
        self.x_downsample_layer = nn.Conv2d(in_ch, out_ch, kernel_size = 2, stride = 2
                    ) # x下采样，使用get_downsample_x2_conv2_layer的下采样方法
        self.yH_upsample_layer = nn.Sequential(
                        nn.Upsample(scale_factor = 2, mode='nearest'),
                        nn.Conv2d(in_ch, in_ch, kernel_size = 3, padding = 1),
                    ) # yH上采样，使用get_upsample_x2_upconv_layer的上采样方法
        
        self.SELayer_yhl = SELayer_2d(channel=in_ch, reduction=16)
        self.SELayer_ylh = SELayer_2d(channel=in_ch, reduction=16)
        self.SELayer_yhh = SELayer_2d(channel=in_ch, reduction=16)

    def forward(self, x):

        ## 提取高、低频信号
        yL, yH = self.wt(x) # yL是tensor，yH是list
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]

        # x_in = self.x_InstanceNorm_layer(x)  # 原特征图归一化x(n,2c,h,w)

        ## 高频增强
        # yH_cat = y_HL + y_LH + y_HH # 高频信号(n,2c,h/2,w/2)

        ## 三高频相加再做通道注意力
        # yH_cat = self.SELayer(yH_cat)

        ## 三高频分别做通道注意力
        y_HL = self.SELayer_yhl(y_HL)
        y_LH = self.SELayer_ylh(y_LH)
        y_HH = self.SELayer_yhh(y_HH)
        yH_cat = y_HL + y_LH + y_HH # 高频信号(n,2c,h/2,w/2)


        yH_cat = self.yH_upsample_layer(yH_cat) # 高频信号(n,2c,h/2,w/2)→(n,2c,h,w)
        # yH_cat = self.yH_InstanceNorm_layer(yH_cat) # 高频信号归一化(n,2c,h/2,w/2)
        x_H = x + yH_cat # 高频增强x(n,2c,h,w)
        # x_H = self.x_H_InstanceNorm_layer(x_H) # 高频增强x归一化(n,2c,h,w)→(n,2c,h,w)

        ## 低频增强
        xds = self.x_downsample_layer(x) # 原特征图x下采样(n,2c,h/2,w/2)
        # xds = self.xds_InstanceNorm_layer(xds) # 下采样特征图归一化(n,2c,h/2,w/2)
        # yL = self.yL_InstanceNorm_layer(yL) # 低频信号归一化(n,2c,h/2,w/2)
        x_L = xds + yL # 低频增强x(n,2c,h/2,w/2)

        return x_H, xds

################################################################################

def get_downsample_x2_conv2_layer(features, **kwargs):
    return (
        nn.Conv2d(features, features, kernel_size = 2, stride = 2, **kwargs),
        features
    )

def get_downsample_x2_conv3_layer(features, **kwargs):
    return (
        nn.Conv2d(
            features, features, kernel_size = 3, stride = 2, padding = 1,
            **kwargs
        ),
        features
    )

def get_downsample_x2_pixelshuffle_layer(features, **kwargs):
    out_features = 4 * features
    return (nn.PixelUnshuffle(downscale_factor = 2, **kwargs), out_features)

def get_downsample_x2_pixelshuffle_conv_layer(features, **kwargs):
    out_features = features * 4

    layer = nn.Sequential(
        nn.PixelUnshuffle(downscale_factor = 2, **kwargs),
        nn.Conv2d(
            out_features, out_features, kernel_size = 3, padding = 1
        ),
    )

    return (layer, out_features)

def get_upsample_x2_deconv2_layer(features, **kwargs):
    return (
        nn.ConvTranspose2d(
            features, features, kernel_size = 2, stride = 2, **kwargs
        ),
        features
    )

def get_upsample_x2_upconv_layer(features, **kwargs):
    layer = nn.Sequential(
        nn.Upsample(scale_factor = 2, **kwargs),
        nn.Conv2d(features, features, kernel_size = 3, padding = 1),
    )

    return (layer, features)

def get_upsample_x2_pixelshuffle_conv_layer(features, **kwargs):
    out_features = features // 4

    layer = nn.Sequential(
        nn.PixelShuffle(upscale_factor = 2, **kwargs),
        nn.Conv2d(out_features, out_features, kernel_size = 3, padding = 1),
    )

    return (layer, out_features)

def get_downsample_x2_layer(layer, features):
    name, kwargs = extract_name_kwargs(layer)

    ######################### 改，添加Haar小波变换下采样方法 ###########################
    if name == 'hwd':
        return (Down_wt(features, features), features)

    ################################################################################
    ######################### 改，添加Haar小波变换频域增强方法 ##########################
    if name == 'wtfd_cat':
        return (wtfd_cat(features, features), features)
    if name == 'wtfd_add':
        return (wtfd_add(features, features), features)
    ################################################################################

    if name == 'conv':
        return get_downsample_x2_conv2_layer(features, **kwargs)

    if name == 'conv3':
        return get_downsample_x2_conv3_layer(features, **kwargs)

    if name == 'avgpool':
        return (nn.AvgPool2d(kernel_size = 2, stride = 2, **kwargs), features)

    if name == 'maxpool':
        return (nn.MaxPool2d(kernel_size = 2, stride = 2, **kwargs), features)

    if name == 'pixel-unshuffle':
        return get_downsample_x2_pixelshuffle_layer(features, **kwargs)

    if name == 'pixel-unshuffle-conv':
        return get_downsample_x2_pixelshuffle_conv_layer(features, **kwargs)

    raise ValueError("Unknown Downsample Layer: '%s'" % name)

def get_upsample_x2_layer(layer, features):
    name, kwargs = extract_name_kwargs(layer)

    if name == 'deconv':
        return get_upsample_x2_deconv2_layer(features, **kwargs)

    if name == 'upsample':
        return (nn.Upsample(scale_factor = 2, **kwargs), features)

    if name == 'upsample-conv':
        return get_upsample_x2_upconv_layer(features, **kwargs)

    if name == 'pixel-shuffle':
        return (nn.PixelShuffle(upscale_factor = 2, **kwargs), features // 4)

    if name == 'pixel-shuffle-conv':
        return get_upsample_x2_pixelshuffle_conv_layer(features, **kwargs)

    raise ValueError("Unknown Upsample Layer: '%s'" % name)

