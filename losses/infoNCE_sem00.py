import sys
sys.path.append('/path/to/vugan')  # 替换为实际路径

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.nn import functional as F
import functools
import math
import clip
from clip.model import ModifiedResNet
# from vugan.modules.module_attention import ModifiedSpatialTransformer
import torchvision.transforms as T
from torch.amp import autocast


class CLIP_Semantic_extractor(ModifiedResNet):
    def __init__(self, layers=(3, 4, 6, 3), pretrained=True, path=None, output_dim=1024, heads=32):
        super(CLIP_Semantic_extractor, self).__init__(layers=layers, output_dim=output_dim, heads=heads)

        ckpt = 'RN50' if path is None else path
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if pretrained:
            model, preprocess = clip.load(ckpt, device=device) # 加载CLIP 模型，（另一个是转换（transform）函数）
        
        self.load_state_dict(model.visual.state_dict()) # 加载预训练模型的视觉编码器部分的权重

        # 注册两个缓冲区 mean 和 std，分别表示图像归一化的均值和标准差（一般来自于 ImageNet 数据集），用于预训练模型的标准化操作
        self.register_buffer(
            'mean',
            torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        )
        self.register_buffer(
            'std',
            torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
        )
        self.requires_grad_(False)

        self.model = model.visual
        del model

    def forward(self, x):
        transform = T.Resize((224, 224)) # 输入图像resize
        x  = transform(x)
        x = (x - self.mean) / self.std # 归一化输入
        x = x.type(self.conv1.weight.dtype) # 令输入张量的数据类型与第一个卷积层的权重相同
        return self.model(x)


'''
# 定义 InfoNCE 损失函数
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, encoded_generated, encoded_real):

        """
        计算InfoNCE损失。
        
        参数:
        - encoded_generated: 生成图像的特征向量，形状为 (batch_size, feature_dim)
        - encoded_real: 真实图像的特征向量，形状为 (batch_size, feature_dim)
        - temperature: 温度参数，默认值为0.07
        
        返回:
        - loss: InfoNCE损失值
        """
        # 对特征进行L2归一化
        encoded_generated = F.normalize(encoded_generated, dim=-1)
        encoded_real = F.normalize(encoded_real, dim=-1)

        batch_size = encoded_generated.size(0)
        feature_dim = encoded_generated.size(-1)

        # 构建正样本对矩阵：每个生成图像与对应的真实图像之间的相似度
        logits_pos = torch.einsum('bd,bd->b', [encoded_generated, encoded_real]).view(-1, 1)

        # 构建负样本对矩阵：每个生成图像与所有其他真实图像之间的相似度
        # 注意避免将自身作为负样本，因此需要构造一个mask来排除这些项
        mask = ~torch.eye(batch_size, device=encoded_generated.device).bool()
        logits_neg = torch.einsum('bd,cd->bc', [encoded_generated, encoded_real])
        logits_neg = logits_neg[mask].view(batch_size, -1)

        # 合并正样本和负样本的logits，并应用温度参数
        logits = torch.cat([logits_pos, logits_neg], dim=1) / self.temperature

        # 标签是全0向量，表示每个样本对应的正样本在logits中的位置
        labels = torch.zeros(batch_size, dtype=torch.long, device=encoded_generated.device)

        # 使用交叉熵损失函数计算InfoNCE损失
        loss = F.cross_entropy(logits, labels)

        return loss
'''
'''
class CLIP_Semantic_extractor(ModifiedResNet):
    def __init__(self, layers=(3, 4, 6, 3), pretrained=True, path=None, output_dim=1024, heads=32):
        super(CLIP_Semantic_extractor, self).__init__(layers=layers, output_dim=output_dim, heads=heads)
        
        # self = self.bfloat16()  # 将整个模型转换为半精度

        ckpt = 'RN50' if path is None else path
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if pretrained:
            model, _ = clip.load(ckpt, device=device) # 加载CLIP 模型，（另一个是转换（transform）函数）
        
        # 将所有权重迁移到设备上
        model.visual = model.visual.to(device)
        self.load_state_dict(model.visual.state_dict()) # 加载预训练模型的视觉编码器部分的权重

        # 注册两个缓冲区 mean 和 std，分别表示图像归一化的均值和标准差（一般来自于 ImageNet 数据集），用于预训练模型的标准化操作
        self.register_buffer(
            'mean',
            torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        )
        self.register_buffer(
            'std',
            torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
        )
        self.requires_grad_(False)

        model.eval()
        del model

    def forward(self, x):

        device = x.device  # 获取模型所在的设备
        # 将所有模型权重迁移到相同设备
        self.to(device)
        with autocast(device_type="cuda", dtype=torch.float32):
            # 确保输入数据类型是 float32
            # x = x.to(torch.float32)

            def stem(x):
                x = self.relu1(self.bn1(self.conv1(x)))
                x = self.relu2(self.bn2(self.conv2(x)))
                x = self.relu3(self.bn3(self.conv3(x)))
                x = self.avgpool(x)
                return x

            x = (x - self.mean.to(device)) / self.std.to(device)
            x = x.type(self.conv1.weight.dtype)
            x = stem(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            return x
'''

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, gen_sem, real_sem):
        """
        计算InfoNCE损失，按批次计算每张图像的InfoNCE，并返回所有图像的平均InfoNCE。
        
        参数:
        - gen_sem: 生成图像的特征，形状为 (batch_size, feature_dim)
        - real_sem: 真实图像的特征，形状为 (batch_size, feature_dim)
        - temperature: 温度参数，默认值为0.07
        
        返回:
        - loss: 本批次的InfoNCE损失的平均值
        """
        batch_size = gen_sem.size(0)
        
        # 对特征进行L2归一化
        gen_sem = F.normalize(gen_sem, dim=-1)
        real_sem = F.normalize(real_sem, dim=-1)

        # 存储所有图像的InfoNCE损失
        all_losses = []

        for i in range(batch_size):
            # 选择当前生成图像的特征和对应的真实图像作为正样本
            gen_i = gen_sem[i].unsqueeze(0)  # 前面添加一个维度： (1, feature_dim)
            real_i = real_sem[i].unsqueeze(0)  # 前面添加一个维度： (1, feature_dim)

            # 正样本对：当前生成图像与其对应真实图像的相似度
            logits_pos = torch.einsum('bd,bd->b', [gen_i, real_i]).view(-1, 1)

            # 负样本对：当前生成图像与其他真实图像的相似度
            mask = torch.ones(batch_size, dtype=torch.bool).to(gen_sem.device)
            mask[i] = False  # 排除自身作为负样本
            real_neg = real_sem[mask]  # 负样本的特征，(batch_size-1, feature_dim * h * w)
            logits_neg = torch.einsum('bd,cd->bc', [gen_i, real_neg]) 
            # 形状 (1, batch_size-1)，每个元素表示生成图像gen_i与一个负样本real_neg[i]的相似度

            # 合并正负样本的logits，并应用温度参数
            logits = torch.cat([logits_pos, logits_neg], dim=1) / self.temperature

            # 标签是全0向量，表示正样本在logits中的位置
            labels = torch.zeros(1, dtype=torch.long).to(gen_sem.device)

            # 计算InfoNCE损失
            loss = F.cross_entropy(logits, labels)
            all_losses.append(loss)

        # 最终返回批次内所有图像的平均InfoNCE损失
        avg_loss = torch.mean(torch.stack(all_losses))

        return avg_loss
        
if __name__ == '__main__':
    clipmodel = CLIP_Semantic_extractor()
    infonce = InfoNCELoss()
    xr = torch.randn(4,1,256,256)
    xg = torch.randn(4,1,256,256)
    xr_s = clipmodel(xr)
    xg_s = clipmodel(xg)
    y = infonce(xg_s, xr_s)

    print(y.size())
    print(y)
