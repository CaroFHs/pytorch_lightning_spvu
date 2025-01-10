import torch
import torch.nn as nn
from torch.autograd import Variable
import yaml


# 读取 YAML 配置文件
config_path = 'config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# 获取模型配置
ganloss_config = config['ganloss']

class GANLoss(nn.Module):
    def __init__(self, gan_type='lsgan', target_real_label=1.0, target_fake_label=0.0,
                 Gganloss_weight=ganloss_config['Gganloss_weight'], 
                 Dganloss_weight=ganloss_config['Dganloss_weight']):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.Gganloss_weight = Gganloss_weight
        self.Dganloss_weight = Dganloss_weight
        self.real_label = target_real_label
        self.fake_label = target_fake_label

        if self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'vanilla':
            self.loss = nn.BCELoss()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def get_target_label(self, input, label):
        target_val = (
            self.real_label if label else self.fake_label)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, label, is_disc=False):

        target_label = self.get_target_label(input, label)
        Gganloss = self.loss(input, target_label)* self.Gganloss_weight
        Dganloss = self.loss(input, target_label)* self.Dganloss_weight

        return Gganloss if is_disc else Dganloss
    
