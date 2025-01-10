# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

import torch
from torch import nn

from vugan.vugan_select import get_norm_layer, get_activ_layer


################################ 生成器为纯VIT时调用，此项目中没有用到 ######################################

def calc_tokenized_size(image_shape, token_size):
    # image_shape : (C, H, W)
    # token_size  : (H_t, W_t)
    if image_shape[1] % token_size[0] != 0: # 检查图像的高度 H 是否能被令牌的高度 H_t 整除
        raise ValueError(
            "Token width %d does not divide image width %d" % (
                token_size[0], image_shape[1]
            )
        ) # 若不能整除则报错

    if image_shape[2] % token_size[1] != 0: # 检查图像的宽度 W 是否能被令牌的宽度 W_t 整除
        raise ValueError(
            "Token height %d does not divide image height %d" % (
                token_size[1], image_shape[2]
            )
        ) # 若不能整除则报错

    # result : (N_h, N_w)
    return (image_shape[1] // token_size[0], image_shape[2] // token_size[1]) # 返回计算出的令牌化后的尺寸

def img_to_tokens(image_batch, token_size):
    '''
    将图像批次转化为令牌化后的图像
    输入：image_batch (N, C, H, W)，token_size (H_t, W_t)
    '''
    # image_batch : (N, C, H, W)
    # token_size  : (H_t, W_t)

    # result : (N, C, N_h, H_t, W)
    result = image_batch.view(
        (*image_batch.shape[:2], -1, token_size[0], image_batch.shape[3])
    ) # 将图像的高度分成令牌块

    # result : (N, C, N_h, H_t, W       )
    #       -> (N, C, N_h, H_t, N_w, W_t)
    result = result.view((*result.shape[:4], -1, token_size[1]))# 将图像的宽度分成令牌块

    # result : (N, C, N_h, H_t, N_w, W_t)
    #       -> (N, N_h, N_w, C, H_t, W_t)
    result = result.permute((0, 2, 4, 1, 3, 5)) # 调整令牌块维度

    return result

def img_from_tokens(tokens):
    '''
    将令牌化的图像块重新组合成原始图像
    '''
    # tokens : (N, N_h, N_w, C, H_t, W_t)
    # result : (N, C, N_h, H_t, N_w, W_t)

    ## 调整令牌块维度:将通道数 C 移动到前面，使得后续的重塑操作更加方便
    result = tokens.permute((0, 3, 1, 4, 2, 5))

    # result : (N, C, N_h, H_t, N_w, W_t)
    #       -> (N, C, N_h, H_t, N_w * W_t)
    #        = (N, C, N_h, H_t, W)
    ## 将每个令牌块的宽度重新组合成图像的宽度
    result = result.reshape((*result.shape[:4], -1))

    # result : (N, C, N_h, H_t, W)
    #       -> (N, C, N_h * H_t, W)
    #        = (N, C, H, W)
    ## 将每个令牌块的高度重新组合成图像的高度
    result = result.reshape((*result.shape[:2], -1, result.shape[4]))

    return result

############################################################################################


class PositionWiseFFN(nn.Module):
    '''
    位置编码的前馈神经网络
    '''
    def __init__(self, features, ffn_features, activ = 'gelu', **kwargs):
        '''
        初始化网络
        输入：features: 输入特征的维度； ffn_features: 前馈网络的隐藏层特征维度；
              activ: 激活函数的名称，默认 'gelu'； kwargs: 其他可选参数，用于传递给父类的构造函数。
        '''
        super().__init__(**kwargs) # 调用父类的构造函数

        self.net = nn.Sequential(
            nn.Linear(features, ffn_features),
            get_activ_layer(activ),
            nn.Linear(ffn_features, features),
        ) # 构建网络：线性层 + 激活函数 + 线性层

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    '''
     Transformer 块
    '''
    def __init__(
        self, features, ffn_features, n_heads, activ = 'gelu', norm = None,
        use_ffn = True, # 改，添加选择使用ffn的opt参数
        rezero = True, **kwargs
    ):
        '''
        初始化Transformer 块
        输入：features: 输入和输出特征的维度； ffn_features: 前馈网络的隐藏层特征维度；
              n_heads: 多头注意力机制中的注意力头数； activ: 激活函数的名称，默认为 'gelu'；
              norm: 归一化层的类型，默认为 None； rezero: 是否使用 ReZero 技术，默认为 True；
              kwargs: 其他可选参数，用于传递给父类的构造函数。
        '''
        super().__init__(**kwargs) # 调用父类

        self.use_ffn = use_ffn # 改，添加选择使用ffn的opt参数

        self.norm1 = get_norm_layer(norm, features) # 归一化层1
        self.atten = nn.MultiheadAttention(features, n_heads) # 多头注意力层

        self.norm2 = get_norm_layer(norm, features) # 归一化层2

        if self.use_ffn == True: # 改，改为使用opt参数判断是否使用ffn
            self.ffn   = PositionWiseFFN(features, ffn_features, activ) # 前馈神经网络 # 改，去掉FFN

        self.rezero = rezero # 使用 ReZero 技术来提高训练的稳定性和速度

        if rezero: # 若使用ReZero
            self.re_alpha = nn.Parameter(torch.zeros((1, ))) # 创建一个可训练参数 self.re_alpha，初始化为零
        else:
            self.re_alpha = 1 # 否则，将 self.re_alpha 设置为 1

    def forward(self, x):
        ''' 前向传播过程 '''
        # x: (L, N, features)

        # Step 1: Multi-Head Self Attention
        y1 = self.norm1(x)
        y1, _atten_weights = self.atten(y1, y1, y1)

        y  = x + self.re_alpha * y1


        if self.use_ffn == True: # 改，改为使用opt参数判断是否使用ffn
    ####################### 原代码 ##########################
        # Step 2: PositionWise Feed Forward Network
            y2 = self.norm2(y)
            y2 = self.ffn(y2)

            y  = y + self.re_alpha * y2
    #########################################################
    ####################### 改，去掉FFN ######################
        if self.use_ffn == False: # 改，改为使用opt参数判断是否使用ffn
            y2 = self.norm2(y)
            y  = y + y2

    #########################################################
        return y

    def extra_repr(self):
        ''' 返回一个字符串，提供了关于该模块的额外信息，通常用于调试和模型打印 '''
        return 're_alpha = %e' % (self.re_alpha, )

class TransformerEncoder(nn.Module):
    '''
    编码器
    '''
    def __init__(
        self, features, ffn_features, n_heads, n_blocks, activ, norm, use_ffn, # 改，添加选择使用ffn的opt参数
        rezero = True, **kwargs
    ):
        ''' 初始化
        输入：features: 输入和输出特征的维度； ffn_features: 前馈网络的隐藏层特征维度；
              n_heads: 多头注意力机制中的注意力头数； n_blocks：transformer块数；
              activ: 激活函数的名称，默认为 'gelu'； norm: 归一化层的类型，默认为 None； 
              rezero: 是否使用 ReZero 技术，默认为 True； kwargs: 其他可选参数，用于传递给父类的构造函数。
        '''
        super().__init__(**kwargs) # 调用父类

        self.encoder = nn.Sequential(*[
            TransformerBlock(
                features, ffn_features, n_heads, activ, norm, use_ffn, # 改，添加选择使用ffn的opt参数
                rezero
            ) for _ in range(n_blocks)
        ]) # 构建编码器：n_blocks个transformer快

    def forward(self, x):
        # x : (N, L, features)

        # y : (L, N, features)
        y = x.permute((1, 0, 2))
        y = self.encoder(y)

        # result : (N, L, features)
        result = y.permute((1, 0, 2))

        return result

class FourierEmbedding(nn.Module):
    # arXiv: 2011.13775
    '''
    基于 Fourier 特征嵌入的模块，用于将图像的空间坐标嵌入到更高维的特征空间中
    '''
    def __init__(self, features, height, width, **kwargs):
        '''
        初始化，输入：
        features: 输出特征的维度； 384
        height: 输入图像的高度；   16
        width: 输入图像的宽度；    16
        **kwargs: 其他可选参数，传递给父类的构造函数
        '''
        super().__init__(**kwargs) # 调用父类
        self.projector = nn.Linear(2, features) # 线性层，将二维坐标投影到指定的特征维度
        self._height   = height # 输入图像的高度
        self._width    = width  # 输入图像的宽度

    def forward(self, y, x):
        # x : (N, L)
        # y : (N, L)
        ## 坐标归一化
        x_norm = 2 * x / (self._width  - 1) - 1
        y_norm = 2 * y / (self._height - 1) - 1

        # z : (N, L, 2) = [1, 256, 2]
        ## 将归一化后的 x 和 y 坐标沿最后一个维度拼接
        z = torch.cat((x_norm.unsqueeze(2), y_norm.unsqueeze(2)), dim = 2)

        ## 将拼接后的坐标通过线性层投影到 features 维度，并对结果应用 sin 函数，得到最终的 Fourier 特征嵌入
        z = torch.sin(self.projector(z)) # [1, 256, 384]
        # print('###########')
        return z

############################################################################################
# class PatchEmbed(nn.Module):
#     """ 
#     Image to Patch Embedding
#     """
#     def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
#         '''
#         图像的默认尺寸:256  假设为正方形，实际通过to_2tuple转换为宽度和高度;
#         每个patch的尺寸:8   样假设为正方形;
#         输入图像的通道数:3  默认为RGB图像的3通道;
#         嵌入维度:512        即每个patch经过处理后输出的特征向量维度
#         '''
#         super().__init__()

#         # [256, 256]
#         img_size = to_2tuple(img_size)     # 将输入的单个整数尺寸转换为包含宽度和高度的元组形式，以确保后续操作的兼容性
#         # [8, 8]
#         patch_size = to_2tuple(patch_size) # 同上
#         # num_patches = (256 // 8) * (256 // 8)
#         num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) # 计算整个图像可以分割成多少个patch
#         self.img_size = img_size # [256, 256]
#         self.patch_size = patch_size # [8, 8]
#         self.num_patches = num_patches # 1024

#       ## 定义一个卷积层proj，作用是将输入图像分割成patch并映射到指定的嵌入维度，
#       ## 卷积核的尺寸和步长都设为patch_size。
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
#       ## 添加一个上采样层up1，使用最近邻插值方式，放大因子为2，
#       ## 这个层在当前的forward方法中未被使用，可能是为后续的模型扩展或修改预留的。
#         self.up1 = nn.Upsample(scale_factor=2, mode='nearest')


#     def forward(self, x):
#         '''
#         前向传播过程
#         '''
#       ## 获取输入张量的形状:(B, C, H, W)，其中B是批量大小，C是通道数，H和W分别是图像的高度和宽度
#         B, C, H, W = x.shape 
#       ## 使用之前定义的卷积层proj处理输入图像x，将图像分割成patch并映射到高维空间
#         x = self.proj(x) # x: [1, 512, 64, 64]

#         return x

###############################################################################


class ViTInput(nn.Module):
    ''' VIT输入 '''
    def __init__(
        self, input_features, embed_features, features, height, width,
        **kwargs
    ):
        '''
        初始化，输入：
        input_features：    输入图像块的特征数； 
        embed_features：    位置嵌入的特征数；
        features：          输出特征数； 
        height 和 width：   图像的高度和宽度；
        **kwargs：          其他额外参数；
        '''
        super().__init__(**kwargs) # 调用父类
        self._height   = height # 16
        self._width    = width  # 16

        ## 生成 0 到 width-1 和 0 到 height-1 的序列
        x = torch.arange(width).to(torch.float32)  # x = [0., 1., ..., 13.]
        y = torch.arange(height).to(torch.float32) # y = [0., 1., ..., 13.]

        x, y   = torch.meshgrid(x, y, indexing='ij') # 创建一个网格，将 x 和 y 坐标结合起来，生成所有像素的位置

        ## x和y的坐标展成一维
        self.x = x.reshape((1, -1))
        self.y = y.reshape((1, -1))

        ## 将 x 和 y 坐标注册为缓冲区（buffer），这意味着它们不会作为参数更新，但会随模型一起保存和加载
        self.register_buffer('x_const', self.x)
        self.register_buffer('y_const', self.y)

        ## 傅里叶嵌入层，将 x 和 y 坐标嵌入到高维特征空间中。
        self.embed  = FourierEmbedding(embed_features, height, width)

        ## 线性输出层，将输入特征和嵌入特征合并，并转换为指定的输出特征数
        self.output = nn.Linear(embed_features + input_features, features)

    def forward(self, x):
        # x     : (N, L, input_features)
        # embed : (1, height * width, embed_features)
        #       = (1, L, embed_features) = [1, 256, 384]
        embed = self.embed(self.y_const, self.x_const)

        # embed : (1,/sd L, embed_features)
        #      -> (N, L, embed_features) = [1, 256, 384]
        embed = embed.expand((x.shape[0], *embed.shape[1:]))

        # result : (N, L, embed_features + input_features) = [1, 256, 768]
        result = torch.cat([embed, x], dim = 2)

        # (N, L, features) = [1, 256, 384]
        output = self.output(result) # 使用nn.Linear()函数，输入个数768，输入384
        return output # [1, 256, 384]

class PixelwiseViT(nn.Module):
    ''' 像素级VIT '''
    def __init__(
        self, features, n_heads, n_blocks, ffn_features, embed_features,
        activ, norm, image_shape, use_ffn, # 改，添加选择使用ffn的opt参数
        rezero = True, **kwargs
    ):
        '''
        features：          Transformer 模型中的特征数； 
        n_heads：           多头自注意力机制中的头数；
        _blocks：           Transformer 编码器中的块数； 
        ffn_features：      前馈神经网络的特征数；
        embed_features：    位置嵌入的特征数； 
        activ：             激活函数类型；
        norm：              归一化类型； 
        image_shape：       输入图像的形状 (C, H, W)；
        rezero：            是否使用 ReZero 技术； 
        **kwargs：          其他额外参数。
        '''
        super().__init__(**kwargs) # 调用父类

        self.image_shape = image_shape # list:[384, 16, 16]

        self.trans_input = ViTInput(
            image_shape[0], embed_features, features,
            image_shape[1], image_shape[2],
        ) # 创建vitinput实例，将输入图像块进行嵌入和位置编码
        # ViTInput(输入图像块的特征数，位置嵌入的特征数，输出特征数，图像高度，图像宽度)

        self.encoder = TransformerEncoder(
            features, ffn_features, n_heads, n_blocks, activ, norm, use_ffn, # 改，添加选择使用ffn的opt参数
            rezero
        ) # 创建Transformer 编码器对嵌入的输入进行编码

        ## 创建线性输出层，将编码后的特征转换为指定的输出特征数
        self.trans_output = nn.Linear(features, image_shape[0])

    def forward(self, x):
        # x : (N, C, H, W) = [1, 384, 16, 16]

    #### pre-transforms
        # itokens : (N, C, H * W) = [1, 384, 256]
        itokens = x.view(*x.shape[:2], -1)

        # itokens : (N, C,     H * W)
        #        -> (N, H * W, C    )
        #         = (N, L,     C) = [1, 256, 384]
        itokens = itokens.permute((0, 2, 1))

    #### patch embedding & pos embedding
        # y : (N, L, features) = [1, 256, 384]
        y = self.trans_input(itokens)

    #### transformer encoder
        y = self.encoder(y) # [1, 256, 384]

    #### linear layer
        # otokens : (N, L, C) = [1, 256, 384]
        otokens = self.trans_output(y)

        # otokens : (N, L, C)
        #        -> (N, C, L)
        #         = (N, C, H * W) = [1, 384, 256]
        otokens = otokens.permute((0, 2, 1))

        # result : (N, C, H, W) = [1, 384, 16, 16]
        result = otokens.view(*otokens.shape[:2], *self.image_shape[1:])

        return result

