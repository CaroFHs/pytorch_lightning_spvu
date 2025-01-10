import torch
import math
import torch.nn as nn
from torch.nn import functional as F


def add_bias_channel(x, dim=1):
    one_size = list(x.size())
    one_size[dim] = 1
    one = x.new_ones(one_size)
    return torch.cat((x, one), dim)


def flatten(x, keepdims=False):
    """
    Flattens B C H W input to B C*H*W output, optionally retains trailing dimensions.
    """
    y = x.view(x.size(0), -1)
    if keepdims:
        for d in range(y.dim(), x.dim()):
            y = y.unsqueeze(-1)
    return y



class GeMPooling2d(nn.Module):
    def __init__(self, p=3, eps=1e-6, kernel_size=3, stride=1, padding=1, clamp=True, add_bias=False, keepdims=False):
        '''
        p:          池化参数
        clamp:      将输入张量中的值限制在一个最小值之上，以确保在进行幂运算时不会出现负数或非常小的数值;
        add_bias:   添加偏置
        keepdims:   保持输入的维度
        '''
        super(GeMPooling2d, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.clamp = clamp
        self.add_bias = add_bias
        self.keepdims = keepdims

    def forward(self, x):
        if self.p == math.inf or self.p is 'inf':
            x = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        elif self.p == 1 and not (torch.is_tensor(self.p) and self.p.requires_grad):
            x = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        else:
            if self.clamp:
                x = x.clamp(min=self.eps)
            x = F.avg_pool2d(x.pow(self.p), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).pow(1.0 / self.p)
        if self.add_bias:
            x = add_bias_channel(x)
        if not self.keepdims:
            x = flatten(x)
        return x

##############################################################################################
##############################################################################################
class In_Conv(nn.Module):
    def __init__(self, img_dim, norm, act):
        super().__init__()
        self.img_in_conv = nn.Sequential(
            nn.Conv2d(3, img_dim, 3, padding = 1, bias = False),
            norm(img_dim),
            act()
        )

    def forward(self, x):
        y = self.img_in_conv(x)
        return y


class MEEM_GMP2d(nn.Module):
    def __init__(self, in_dim, hidden_dim, width, norm, act):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias = False),
            norm(hidden_dim),
            nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride= 1, padding = 1)
        # self.pool = GeMPooling2d(p=3, kernel_size=3, stride=1, padding=1, clamp=True, add_bias=False, keepdims=True)

        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()
        for i in range(width - 1):
            self.mid_conv.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias = False),
                norm(hidden_dim),
                nn.Sigmoid()
            ))
            self.edge_enhance.append(EdgeEnhancer(hidden_dim, norm, act))

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * width, in_dim, 1, bias = False),
            norm(in_dim),
            act()
        )
    
    def forward(self, x):
        mid = self.in_conv(x) # (4,16,h,w)

        out = mid
        #print(out.shape)
        
        for i in range(self.width - 1):
            # (4,16,h,w) → (4,32,h,w) → (4,48,h,w) → (4,64,h,w)
            mid = self.pool(mid) # (4,16,h,w)
            mid = self.mid_conv[i](mid) # (4,16,h,w)

            out = torch.cat([out, self.edge_enhance[i](mid)], dim = 1)
        
        out = self.out_conv(out) # (4,64,h,w) → (4,32,h,w)

        return out # Fme (4,32,h,w)

class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias = False),
            norm(in_dim),
            nn.Sigmoid()
        )
        self.pool = GeMPooling2d(p=3, kernel_size=3, stride=1, padding=1, clamp=True, add_bias=False, keepdims=True)
    
    def forward(self, x):
        # x (4,16,h,w)
        edge = self.pool(x) # (4,16,h,w)
        edge = x - edge # (4,16,h,w)
        edge = self.out_conv(edge) # (4,16,h,w)
        # return x + edge
        return edge
    

class DE(nn.Module):
    def __init__(self, c_in, img_dim, norm, act):
        '''
        c_in：输入通道数；
        img_dim：输出通道数；
        norm：归一化方法；
        act：激活函数
        '''
        super().__init__()
        self.img_in_conv = nn.Sequential(
            nn.Conv2d(c_in, img_dim, 3, padding = 1, bias = False),
            norm(img_dim),
            act()
        )
        self.meem = MEEM_GMP2d(img_dim, img_dim  // 2, 4, norm, act)
    
    def forward(self, x):
        x_in = self.img_in_conv(x) # Flocal(4, 32, h, w)
        y = self.meem(x_in) # Fme(4, 32, h, w)
        # out = y + x_in # 细节增强后的特征图 + 原特征图
        return y # (4, 32, h, w)
    



if __name__ == '__main__':

    x = torch.randn(4,3,512,512)
    b,c,h,w = x.shape

    # avgpool = nn.AvgPool2d(3, stride= 1, padding = 1)
    # gempool = GeMPooling2d(p=3, eps=1e-6, clamp=True, add_bias=False, keepdims=True)
    de = DE(c_in=c, img_dim=48, norm=nn.BatchNorm2d, act=nn.ReLU)
    y = de(x)

    print('x_shape:', x.shape)
    print('y_shape:', y.shape)

    # 打印模型参数量
    total_params = sum(p.numel() for p in de.parameters())
    print(f'Total number of parameters: {total_params}') # 11040