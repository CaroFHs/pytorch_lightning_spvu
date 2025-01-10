import torch
import torch.nn as nn
import torchvision.models as models



def calc_mean_std(feat, eps=1e-5):
    '''
    计算特征图的均值和标准差，输入为特征图，形状为(N, C, H, W)
    输出为特征图的均值和标准差，形状分别为(N, C, 1, 1)和(N, C, 1)
    '''
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def normal(feat, eps=1e-5):
    '''
    功能：基于calc_mean_std对特征图进行归一化。
    输入：feat - 特征图; eps - 同上。
    输出：归一化后的特征图。
    '''
    feat_mean, feat_std= calc_mean_std(feat, eps)
    normalized=(feat-feat_mean)/feat_std
    return normalized 


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()

        ## 创建一次 VGG16 模型，并指定使用预训练权重
        vgg_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        enc_layers = list(vgg_model.features.children()) # 通过.children()方法将encoder模型拆分成多个子模块

      ## 提取 VGG16 的特征层，并设置为评估模式

        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:9])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[9:16])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[16:23])  # relu3_1 -> relu4_1
        self.enc_1.eval()
        self.enc_2.eval()
        self.enc_3.eval()
        self.enc_4.eval()

        ## 冻结参数
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
        
        self.transform = torch.nn.functional.interpolate # 定义插值函数
        self.resize = resize # 标志位，用于指示是否需要在计算特征之前调整输入图像的大小

        # self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.mse_loss = nn.MSELoss()     # 均方误差损失


    def encode_with_intermediate(self, input):
        '''
        通过一个预定义的编码器网络（由enc_1到enc_5表示）处理输入，并收集并返回这些层的输出
        '''
        results = [input]  # 初始化列表results，将输入数据input作为第一项
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1)) # 获取enc_1到enc_5的输出
          ## 将results列表中的最后一个元素（即上一次迭代的输出或原始输入）作为参数传递，然后将方法的输出添加到列表中
            results.append(func(results[-1]))
        return results[1:] # 返回results列表（除了第一个元素，即原始输入）的所有元素（enc_1到enc_4的输出），即中间结果

    def calc_content_loss(self, input, target):
        '''
        计算内容损失（Content Loss），即输入图像（input）与目标图像（target）在内容上的差异
        '''
        assert (input.size() == target.size()) # 判断确保输入(input)和目标(target)张量（tensors）的大小相同，
        assert (target.requires_grad is False) # 确保目标图像不需要计算梯度，因为目标图像是固定的，不参与反向传播
        return self.mse_loss(input, target)    # 内容损失 = 均方误差损失

    def calc_style_loss(self, input, target):
        '''
        计算风格损失（Style Loss），即输入图像（input）与目标图像（target）在风格上的差异
        '''
        assert (input.size() == target.size()) # 判断确保输入(input)和目标(target)张量（tensors）的大小相同，
        assert (target.requires_grad is False) # 确保目标图像不需要计算梯度，因为目标图像是固定的，不参与反向传播
        input_mean, input_std = calc_mean_std(input)    # 计算输入(input)的均值和标准差
        target_mean, target_std = calc_mean_std(target) # 计算目标(target)的均值和标准差
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)     # 风格损失 = 均值MSE损失+标准差MSE损失

    
    def forward(self, input, target):

        ## 检查并调整输入图像的通道数
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        ## 输入图像归一化
        # input = normal(input)
        # target = normal(target)

        ## 调整输入图像大小到vgg的输入尺寸224
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        input_feats = self.encode_with_intermediate(input)   # 原红外图特征
        target_feats = self.encode_with_intermediate(target) # 生成红外图特征


        ## 初始化loss
        # vgg_loss = 0.0
        # vgg_loss_c = 0.0
        # vgg_loss_s = 0.0


      ## 内容损失 content loss（生成图像Ics和内容图像之间的MSE，分别取二者的特征图的 最后一层(归一化后) 和 倒数第二层(归一化后) 相加）
        vgg_loss_c = self.calc_content_loss(normal(input_feats[-1]), normal(target_feats[-1])) + self.calc_content_loss(normal(input_feats[-2]), normal(target_feats[-2]))
      ## 风格损失 Style loss（生成图像Ics和风格图像，VGG提取特征，先取原始输入，再循环取所有特征层(1到5)，计算均值和标准差的MSEloss，再将所有loss结果相加）
        vgg_loss_s = self.calc_style_loss(input_feats[0], target_feats[0])
        for i in range(1, 4):
            vgg_loss_s += self.calc_style_loss(input_feats[i], target_feats[i])
        vgg_loss = vgg_loss_c + vgg_loss_s
        # print(vgg_loss, vgg_loss_c, vgg_loss_s) # 调试用，打印loss值
        return vgg_loss
    

if __name__ == '__main__':

    # 创建 VGGPerceptualLoss 实例
    perceptual_loss = VGGPerceptualLoss()
    input = torch.randn(4, 3, 512, 512)  # 示例输入图像
    target = torch.randn(4, 3, 512, 512)  # 示例目标图像

    # 计算感知损失
    loss = perceptual_loss(input, target)
    print("Perceptual Loss:", loss.item())
