import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class SELayer(nn.Module):
    ''' 通道注意力，用于频谱滤波器Spectrum Filter (SF) '''
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Frequency_Spectrum_Dynamic_Aggregation(nn.Module):
    ''' 频谱动态融合FSDA '''
    def __init__(self, nc):
        super(Frequency_Spectrum_Dynamic_Aggregation, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            SELayer(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            SELayer(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0))

    def forward(self, x):
        ori_mag = torch.abs(x)
        ori_pha = torch.angle(x)
        # print('mag_size:',ori_mag.size())
        # print('pha_size:',ori_pha.size())
        mag = self.processmag(ori_mag)
        mag = ori_mag + mag
        pha = self.processpha(ori_pha)
        pha = ori_pha + pha
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        return x_out


class Frequency_Feature_Dynamic_Fusion(nn.Module):
    def __init__(self, nc):
        super(Frequency_Feature_Dynamic_Fusion, self).__init__()
        self.fsda = Frequency_Spectrum_Dynamic_Aggregation(nc)

    def forward(self, x):

        b,c,h,w = x.shape # input, x (b,c,h,w)
        x_freq = torch.fft.rfft2(x, norm='backward') # FFT，x → x_freq，(b, c, h, w//2+1)
        x_freq_fsda = self.fsda(x_freq) # FSDA频域处理，(b, c, h, w//2+1)
        x_freq_spatial = torch.fft.irfft2(x_freq_fsda, s=(h, w), norm='backward') # iFFT，(b,c,h,w)
        
        return x_freq_spatial
    


if __name__ == '__main__':

    x = torch.randn(4,48,256,256)
    b,c,h,w = x.shape
    ffdf = Frequency_Feature_Dynamic_Fusion(nc=c)
    y = ffdf(x)
    print('input_size:',x.size())
    print('output_size:',y.size())


    # x_freq = torch.fft.rfft2(x, norm='backward') # FFT，x → x_freq，(4,48,256,129)
    # # 实数序列的傅里叶变换具有共轭对称性，只需要存储前半部分的频率分量即可，即 N//2+1 的复数序列
    # fsda = Frequency_Spectrum_Dynamic_Aggregation(c)
    # x_freq_fsda = fsda(x_freq) # FSDA频域处理，(4,48,256,129)
    # x_freq_spatial = torch.fft.irfft2(x_freq_fsda, s=(h, w), norm='backward') # iFFT，(4,48,256,256)

    # print('input_size:',x.size())
    # print('x_freq_size:',x_freq.size())
    # print('x_freq_fsda_size:',x_freq_fsda.size())
    # print('output_size:',x_freq_spatial.size())

    # 打印模型参数量
    total_params = sum(p.numel() for p in ffdf.parameters())
    print(f'Total number of parameters: {total_params}') # 9984
