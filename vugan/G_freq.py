import torch.nn.functional as F
import torch
import cv2
import sys

import torch.nn as nn



sys.path.append("..")




#######################################################################################


class KernelSizePredictor(nn.Module):
    ''' 定义高斯卷积核大小预测器 '''

    def __init__(self, feature_channel):
        super(KernelSizePredictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(feature_channel, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        # 预测高斯卷积核大小
        kernel_sizes = torch.relu(self.predictor(x)) * 20 + 1  # 预测的大小范围在 [1, 21]
        return kernel_sizes

class Adaptive_GConv(nn.Module):
    ''' 定义自适应高斯卷积层 '''

    def __init__(self, kernel_size_pre):
        super(Adaptive_GConv, self).__init__()
        self.kernel_size_pre = KernelSizePredictor()

    def forward(self, x, gauss_kernel):
        # 预测卷积核大小
        kernel_sizes = self.kernel_size_pre(x)
        
        # 动态裁剪或填充高斯卷积核
        gaussian_results = [] # 储存每个样本的高斯滤波结果
        for i in range(x.size(0)):
            f_map = x[i]
            kernel_size = int(kernel_sizes[i].item()) # 每个样本对应的卷积核大小
            padding = (kernel_size - 1) // 2

            f_map = F.pad(f_map, pad=(padding,padding,padding,padding),mode='reflect')
            result = F.conv2d(x[i:i+1], padded_kernel, stride=1, padding=padding, groups=x.size(1))
            gaussian_results.append(result)
        
        # 合并结果
        gaussian_results = torch.cat(gaussian_results, dim=0)
        
        return gaussian_results
    
class FrequencyEnhancementModule(nn.Module):
    ''' 定义频率增强模块 '''

    def __init__(self, in_channels):
        super(FrequencyEnhancementModule, self).__init__()
        self.kernel_size_pre = KernelSizePredictor(in_channels)
        self.gaussian_conv = AdaptiveGaussianConvLayer(self.kernel_size_pre)
        self.norm = nn.InstanceNorm2d(in_channels)  # 归一化层

    def forward(self, x, gauss_kernel):
        # 提取低频特征
        low_freq = self.gaussian_conv(x, gauss_kernel)
        
        # 与原特征图相加
        enhanced_features = x + low_freq
        
        # 归一化
        normalized_features = self.norm(enhanced_features)
        
        return normalized_features
    

if __name__ == '__main__':

    x = torch.randn((4,48,512,512))
    kernel_sizes_pre = KernelSizePredictor(x.size())
    kernel_sizes = kernel_sizes_pre(x)
    print(kernel_sizes_pre)
    print(kernel_sizes.shape)