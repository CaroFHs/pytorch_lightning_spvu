import torch
from torch import nn
import yaml
import torchmetrics
from typing import Dict, Any, Optional
from lpips import LPIPS


# 读取 YAML 配置文件, 获取模型配置
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
metrics_config = config['metrics']



class L1(torchmetrics.MeanAbsoluteError):
    def __init__(self):
        super().__init__()

class SSIM(torchmetrics.image.StructuralSimilarityIndexMeasure):
    def __init__(self, data_range=1.0):
        super().__init__(data_range=data_range)

class mSSIM(torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure):
    def __init__(self, data_range=1.0):
        super().__init__(data_range=data_range)

class PSNR(torchmetrics.image.PeakSignalNoiseRatio):
    def __init__(self, data_range=1.0):
        super().__init__(data_range=data_range)

class LPIPS(torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity):
    def __init__(self, net_type='alex'):
        super().__init__(net_type=net_type)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # 判断输入图像通道数
        if preds.shape[1] == 1:
            preds_3c = preds.repeat(1, 3, 1, 1)
            target_3c = target.repeat(1, 3, 1, 1)
        else:
            preds_3c, target_3c = preds, target
        super().update(preds_3c, target_3c)


class MetricManager:
    def __init__(self, config_path: str, device: torch.device):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # 初始化所有可能的损失函数
        self.metrics = {
            "L1": L1(),
            "SSIM": SSIM(data_range=1.0),
            "mSSIM": mSSIM(data_range=1.0),
            "PSNR": PSNR(data_range=1.0),
            "LPIPS": LPIPS(net_type='alex'),
        }

        # 确保所有指标实例都被移动到正确的设备上
        for metric_name, metric in self.metrics.items():
            self.metrics[metric_name] = metric.to(device)

        # 根据配置文件中的权重选择性地激活损失函数
        self.active_metrics = {}
        for metrics_name, use_metrics in self.config.get('metrics', {}).items():
            if use_metrics == True:
                if metrics_name in self.metrics:
                    self.active_metrics[metrics_name] = self.metrics[metrics_name]
                else:
                    raise ValueError(f"Unknown metric {metrics_name} specified in the config.")

    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor, prefix: str) -> Dict[str, torch.Tensor]:
        """计算所有激活的损失函数并返回结果"""
        metrics = {}
        for name, metric_fn in self.active_metrics.items():
            outputs = outputs.to(metric_fn.device).to(torch.float32)
            targets = targets.to(metric_fn.device).to(torch.float32)
            metric_fn.update(outputs.clone(), targets.clone()) # 更新状态
            metric_value = metric_fn.compute()  # 计算最终值
            prefixed_name = f"{prefix}_{name}"
            metrics[prefixed_name] = metric_value
        
        return metrics
