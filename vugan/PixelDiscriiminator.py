import torch
import torch.nn as nn
import functools
import torch.nn.init as init


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.InstanceNorm2d, use_sigmoid=False, gpu_ids=[]):
        '''
        input_nc: 输入通道数。
        ndf: 第一层卷积的过滤器数量，默认为64。
        norm_layer: 归一化层，默认为 nn.BatchNorm2d。
        use_sigmoid: 判别器输出是否使用Sigmoid激活函数。
        gpu_ids: 指定使用哪些GPU。
        '''
        super(PixelDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        ## 判别器主要结构：conv + LeakyReLU + conv + LeakyReLU + conv
        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid: # 若use_sigmoid 为 True，则在最后添加一个 Sigmoid 激活函数
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

        self.initialize_weights(self.net, init_type='kaiming')

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.net, input, self.gpu_ids)
        else:
            return self.net(input)
        
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


if __name__ == '__main__':
    x = torch.randn(4,1,512,512)
    pixle_d = PixelDiscriminator(input_nc=1,gpu_ids=[0])
    y = pixle_d(x)
    print(pixle_d)
    for name, param in pixle_d.named_parameters():
        print(name, param.size())
    print(y.size())
