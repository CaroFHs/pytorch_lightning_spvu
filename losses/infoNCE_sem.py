import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.nn import functional as F
import functools
import math
import clip
from clip.model import ModifiedResNet
from vugan.modules.module_attention import ModifiedSpatialTransformer



class CLIP_Semantic_extractor(ModifiedResNet):
    def __init__(self, layers=(3, 4, 6, 3), pretrained=True, path=None, output_dim=1024, heads=32):
        super(CLIP_Semantic_extractor, self).__init__(layers=layers, output_dim=output_dim, heads=heads)

        ckpt = 'RN50' if path is None else path

        if pretrained:
            model, _ = clip.load(ckpt, device='cpu') # 加载CLIP 模型，（另一个是转换（transform）函数）
        
        self.load_state_dict(model.visual.state_dict()) # 加载预训练模型的视觉编码器部分的权重

        # 注册两个缓冲区 mean 和 std，分别表示图像归一化的均值和标准差（一般来自于 ImageNet 数据集），用于预训练模型的标准化操作
        self.register_buffer(
            'mean',
            torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )
        self.requires_grad_(False)

        self.model = model.visual
        del model

    def forward(self, x):
        x = (x - self.mean) / self.std # 归一化输入
        x = x.type(self.conv1.weight.dtype) # 令输入张量的数据类型与第一个卷积层的权重相同
        return self.model(x)

if __name__ == '__main__':
    clipmodel = CLIP_Semantic_extractor()
    x = torch.randn(4,1,128,128)
    y = clipmodel(x)
    print(y.size())
    print(y)
