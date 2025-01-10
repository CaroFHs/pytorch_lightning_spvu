# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

from torch import nn
import torch.nn.init as init


# from models.vugan.transformer_cape import PixelwiseViT
# from models.vugan.unet00          import UNet
from vugan.vugan_select        import get_activ_layer


class ViTUNetGenerator(nn.Module):
    ''' VITUNet生成器 '''

    def __init__(
        self, features, n_heads, n_blocks, ffn_features, embed_features,
        activ, vit_norm, image_shape, input_nc, output_nc, input_layer_out, # 改，添加输入层输出通道数参数
        unet_features_list, unet_activ, unet_norm,

        unet_downsample, # 改，去掉='conv'，改为由传入opt参数指定unet下采样方法
        use_ffn, # 改，添加选择使用ffn的opt参数
        decoder_use_FHAB, # 改，添加decoder中选择使用FHAB
        encoder_use_FFDF, # 改，添加decoder中选择使用FFDF
        use_dropout, # 改，添加unet_vit中选择使用dropout

        unet_upsample   = 'upsample-conv',
        unet_rezero     = False,
        rezero          = True,
        activ_output    = None,
        which_vit       = None,
        **kwargs
    ):
        '''
        初始化生成器
        参数：features：编码器特征维度；  n_heads：注意力头数；  n_blocks：编码器块数；
              ffn_features：前馈网络特征维度；  embed_features：嵌入特征的维度；
              activ：激活函数；  norm：归一化层类型；  image_shape: 输入图像的形状；
              unet_features_list：UNet各层特征数量列表；  unet_activ：U-Net 使用的激活函数；
              unet_norm：U-Net 使用的归一化层； 

              unet_downsample：U-Net 的下采样方法；  unet_downsample：U-Net 的上采样方法；
              unet_rezero：UNet是否使用 ReZero；  rezero: 是否使用 ReZero 技术；
              activ_output：输出层的激活函数；  

              **kwargs: 其他可选参数
        '''
        # pylint: disable = too-many-locals
        super().__init__(**kwargs) # 调用父类的构造函数进行初始化

        ####################### 判断加载哪种vit #########################
        if which_vit == 'unet_vit':
            from vugan.unet00 import UNet
            from vugan.transformer00 import PixelwiseViT
        elif which_vit == 'unet_vit_cape':
            from vugan.unet00 import UNet
            from vugan.transformer_cape import PixelwiseViT
        elif which_vit == 'unet_deconv_vit':
            from vugan.unet_deconv import UNet
            from vugan.transformer00 import PixelwiseViT
        print('###########################################################')
        print(which_vit)
        

        '''
        if which_vit == 'unet_vit':
            from models.vugan.unet00 import UNet
        elif which_vit ==  'unet_vit_GLSA':
            from models.vugan.unet import UNet
        print('###########################################################')
        print(which_vit)
        '''


        ###############################################################

        self.image_shape = image_shape # 存储图像形状
        self.output_nc   = output_nc # 改，增加输出通道参数，不与输入通道共用参数image_shape[0]


        self.net = UNet(
            unet_features_list, unet_activ, unet_norm, image_shape, 
            input_layer_out, # 改，添加输入层输出通道数
            decoder_use_FHAB, # 改，添加decoder中选择使用FHAB
            encoder_use_FFDF, # 改，添加decoder中选择使用FFDF
            use_dropout, # 改，添加unet_vit中选择使用dropout
            input_nc, # 改，添加输入层的输入通道数input_nc
            output_nc, unet_downsample, unet_upsample, unet_rezero
        ) # 初始化UNet模型

        bottleneck = PixelwiseViT(
            features, n_heads, n_blocks, ffn_features, embed_features,
            activ, vit_norm,
            image_shape = self.net.get_inner_shape(),
            use_ffn = use_ffn, # 改，添加选择使用ffn的opt参数
            rezero      = rezero
        ) # 初始化PixelwiseViT 模型，作为 U-Net 的瓶颈层

        self.net.set_bottleneck(bottleneck) # PixelwiseViT 作为瓶颈层插入到 U-Net 模型中

        self.output = get_activ_layer(activ_output) # 初始化输出层的激活函数


        self.initialize_weights(self.net, init_type='xavier')

    def forward(self, x):
        # x : (N, C, H, W)
        result = self.net(x)

        return self.output(result)

    def initialize_weights(self, net, init_type='kaiming', gain=0.02):
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
            elif classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        net.apply(init_func)