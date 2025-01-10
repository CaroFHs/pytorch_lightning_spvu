import torch
from torch import nn

from losses.ssim import MSSIM, SSIM
from losses.vggLoss import  VGGPerceptualLoss
from losses.freq_loss import  hwt_loss

import yaml
from pytorch_lightning.core import LightningModule
from typing import Dict, Any, Optional

# 读取 YAML 配置文件, 获取模型配置
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
loss_config = config['loss']


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.l1loss = nn.L1Loss()

    def forward(self, source, fake, real):
        return self.l1loss(fake, real)

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.ssim = SSIM()

    def forward(self, source, fake, real):
        return (1 - self.ssim(fake, real))

class MSSIMLoss(nn.Module):
    def __init__(self):
        super(MSSIMLoss, self).__init__()
        self.mssim = MSSIM(channel=1)

    def forward(self, source, fake, real):
        return (1 - self.mssim(fake, real))

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 获取当前可用的设备
        self.VGGLoss = VGGPerceptualLoss().to(device)

    def forward(self, source, fake, real):
        real_detached = real.detach()
        return self.VGGLoss(fake, real_detached)
    
class HWTLoss(nn.Module):
    def __init__(self):
        super(HWTLoss, self).__init__()
        self.hw_loss = hwt_loss()
    def forward(self, source, fake, real):
        return self.hw_loss(fake, real)
    

class LossManager:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # 初始化所有可能的损失函数
        self.losses = {
            "L1_loss": L1Loss(),
            "SSIM_loss": SSIMLoss(),
            "mSSIM_loss": MSSIMLoss(),
            "vgg_loss": VGGLoss(),
            "hwt_loss": HWTLoss(),
            # 添加其他自定义损失函数
        }

        # 根据配置文件中的权重选择性地激活损失函数
        self.active_losses = {}
        for loss_name, loss_weight in self.config.get('loss', {}).items():
            if loss_weight != 0:
                if loss_name in self.losses:
                    self.active_losses[loss_name] = {'fn': self.losses[loss_name], 'weight': loss_weight}
                else:
                    raise ValueError(f"Unknown loss function {loss_name} specified in the config.")
                
    def compute_losses(self, source: torch.Tensor, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算所有激活的损失函数并返回结果"""
        losses = {}
        total_loss = 0
        for name, params in self.active_losses.items():
            loss_value = params['fn'](source.clone(),outputs.clone(), targets.clone()) # 克隆张量以避免原地修改
            weighted_loss = loss_value * params['weight']
            losses[name] = weighted_loss
            total_loss += weighted_loss
        
        losses['total_loss'] = total_loss
        return losses

                    