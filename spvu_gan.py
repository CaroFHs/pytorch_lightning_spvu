import os
from typing import Any, Callable
from pytorch_lightning.core.optimizer import LightningOptimizer
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import torchvision
import torch.nn.init as init
import torch.nn.functional as F


from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.amp import GradScaler, autocast

import yaml
from vugan.sed import CLIP_Semantic_extractor
from TransUNet import TransUNet_G, DiscriminatorManager
from dataset.thermal_dataset import CustomDatasetDataLoader
from losses.GANLoss import GANLoss
from losses.loss import LossManager
from metrics import MetricManager
from set_seed import setup_system



# 读取 YAML 配置文件
config_path = 'config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# 获取模型配置
model_config = config['model']
training_config = config['training']
data_config = config['data']
G_config = config['Generator']
D_config = config['Discriminator']


def initialize_weights(net, init_type='kaiming', gain=0.02):
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


class Generator(nn.Module):
    def __init__(self, which_model_netG, image_shape):
        super(Generator, self).__init__()

        if which_model_netG == 'unet_vit':
            self.netG = TransUNet_G(
                ## ViT
                features = G_config['input_layer_output']*8, 
                n_heads = 6, 
                n_blocks = 6, 
                ffn_features = G_config['input_layer_output']*8*2,
                embed_features = G_config['input_layer_output']*8,
                vit_activ = G_config['vit_activ'], 
                vit_norm = G_config['vit_norm'], 
                use_ffn = G_config['vit_use_ffn'], # vit是否使用ffn
                rezero = True,
                ## UNet
                image_shape = image_shape, 
                input_nc = 3,  # 输入层输入通道数
                output_nc = 1, # 输出层输出通道数
                input_layer_out = G_config['input_layer_output'], # 改，添加输入层输出通道数参数
                unet_features_list = [48, 96, 192, 384],  # unet每层输出特征图通道数

                unet_activ = G_config['unet_activ'], 
                unet_norm = G_config['unet_norm'],
                unet_downsample = G_config['unet_downsample'], # unet下采样方法
                unet_upsample   = 'upsample-conv', # unet上采样方法

                decoder_use_FHAB = G_config['decoder_use_FHAB'], # decoder是否使用FHAB
                encoder_use_FFDF = G_config['encoder_use_FFDF'], # decoder是否使用FFDF
                unet_use_dropout = G_config['unet_use_dropout'], # unet是否使用dropout
                unet_rezero     = False,
                activ_output    = None 
            )
        else:
            raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)

    def forward(self, x):
        y = self.netG(x)
        return y


class Discriminator(nn.Module):
    def __init__(self, which_model_netD):
        super(Discriminator, self).__init__()
        self.netD = DiscriminatorManager(
            which_model_netD,
            input_nc = model_config['output_nc'],   # 输入层输入通道数
            # output_nc = model_config['output_nc'], # 输出层输入通道数
            ndf = G_config['input_layer_output'],  # 第一层卷积输入通道数
            use_dropout = G_config['unet_use_dropout'], # unet是否使用dropout
            resolution = model_config['output_size'],  # 输入分辨率
            )
        self.which_model_netD = which_model_netD

    def forward(self, x, semantic):
        if self.which_model_netD in {'sed_p', 'sed_u'}:
            return self.netD(x, semantic)
        else:
            return self.netD(x)
    
class GAN(LightningModule):

    def __init__(self,
                n_critic=5,  # 每n_critic次判别器训练后训练一次生成器
                lr=training_config['lr'],
                b1=training_config['b1'],
                b2=training_config['b2'],
                batch_size=training_config['batch_size'],
                image_shape=G_config['image_shape'],  # 输入图像的形状
                 **kwargs):
        super().__init__()
        self.n_critic = n_critic
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.save_hyperparameters()

        # networks
        self.generator = Generator(
            which_model_netG=G_config['which_model_netG'],
            image_shape=self.image_shape
        )
        self.discriminator = Discriminator(
            which_model_netD=D_config['which_model_netD']
        )
        # print(self.discriminator)
        self.sem_extractor = CLIP_Semantic_extractor()
        for param in self.sem_extractor.parameters():
            param.requires_grad = False
        self.sem_extractor.eval()

        # optimizer
        self.optimizer_type = training_config['optimizer_type']

        # loss
        self.loss_manager = LossManager(config_path)
        self.scaler = GradScaler() # 改

        # metrics
        self.metric_manager_train = MetricManager(config_path, self.device)
        self.metric_manager_val = MetricManager(config_path, self.device)
        self.metric_manager_test = MetricManager(config_path, self.device)

    def forward(self, imgs):
        return self.generator(imgs)
    
    def adversarial_loss(self, label_pre, label, is_disc, gan_type=model_config['gan_type']):
        ganloss = GANLoss(gan_type)
        ad_loss = ganloss(input=label_pre, label=label, is_disc=is_disc)
        return ad_loss
    
    def training_step(self, batch, batch_idx):

        # 读取图像
        source_imgs = batch['A']
        target_imgs = batch['B']
        if torch.isnan(source_imgs).any() or torch.isinf(source_imgs).any():
            raise ValueError("Source_imgs input contains NaN or Inf values")
        if torch.isnan(target_imgs).any() or torch.isinf(target_imgs).any():
            raise ValueError("Target_imgs nput contains NaN or Inf values")

        # 语义提取
        real_semantic = self.sem_extractor(target_imgs)

    #### train generator
        if batch_idx % self.n_critic == 0:
            generated_imgs = self(source_imgs)
            if torch.isnan(generated_imgs).any() or torch.isinf(generated_imgs).any():
                raise ValueError("Detected NaN or Inf in generated images.")
            
            # label_g = torch.ones(generated_imgs.size(0), 1)
            # label_g = label_g.type_as(generated_imgs)
            fake_g_pred = self.discriminator(generated_imgs, real_semantic) # 判别器输出
            Ggan_loss =  self.adversarial_loss(fake_g_pred, True, is_disc=False)# 生成器gan损失
            gen_losses = self.loss_manager.compute_losses(source_imgs, generated_imgs, target_imgs) # 生成损失
            total_g_loss = Ggan_loss + gen_losses['total_loss'] # 总损失
            total_g_loss = gen_losses['total_loss'] # 总损失

            # 立即检查梯度中的 NaN 或 Inf
            for name, param in self.generator.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"NaN or Inf found in {name}'s gradient!")
                    print(f"Gradient: {param.grad}")
                    # raise ValueError(f"Detected NaN or Inf in {name}'s gradient")
                    skip_step = True
                    break

            # 记录生成器阶段损失
            self.log("g_loss", Ggan_loss, prog_bar=True, batch_size=source_imgs.size(0))
            for name, value in gen_losses.items():
                self.log(name, value, prog_bar=True, batch_size=source_imgs.size(0))

            # # 计算训练阶段的评价指标
            # train_metrics = self.metric_manager_train.compute_metrics(generated_imgs, target_imgs, prefix="train")
            # for name, value in train_metrics.items():
            #     self.log(name, value, on_step=False, on_epoch=True, prog_bar=True, batch_size=source_imgs.size(0))

            return {'loss': total_g_loss}

        # train discriminator
        else:
            # 判别器优化
            with torch.no_grad():
                generated_imgs = self(source_imgs).detach().clone()

            # 对于真实图像
            # real = torch.ones(target_imgs.size(0), 1) # 创建一个与输入图片 imgs 批量大小一致的全 1 张量，作为真实标签
            # real = real.type_as(target_imgs) # 确保 valid 的数据类型与输入图片一致
            real_d_pred = self.discriminator(target_imgs, real_semantic) # 判别器输出
            real_loss = self.adversarial_loss(real_d_pred, True, is_disc=True) # 判别器对真实样本 imgs 的预测结果与真实标签 valid 之间的损失

            # 对于生成图像
            # fake = torch.zeros(generated_imgs.size(0), 1)
            # fake = fake.type_as(generated_imgs)
            fake_d_pred = self.discriminator(generated_imgs, real_semantic) # 判别器输出
            fake_loss = self.adversarial_loss(fake_d_pred, False, is_disc=True)

            # 判别器损失
            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True, batch_size=source_imgs.size(0))

            return d_loss

    def validation_step(self, batch, batch_idx):
        source_imgs = batch['A']
        target_imgs = batch['B']

        with torch.no_grad():
            generated_imgs = self(source_imgs)

            # 记录生成图像
            sample_imgs = generated_imgs[:3]
            grid = torchvision.utils.make_grid(tensor=sample_imgs, 
                                            nrow=3, normalize=True, value_range=(0,1), scale_each=True)
            self.logger.experiment.add_image('img_gen', grid, self.current_epoch) # 原为self.current_epoch
            # 记录真实图像
            targ_imgs = target_imgs[:3]
            grid = torchvision.utils.make_grid(tensor=targ_imgs, 
                                            nrow=3, normalize=True, value_range=(0,1), scale_each=True)
            self.logger.experiment.add_image('img_real', grid, self.current_epoch)

            # real_validity = self.discriminator(target_imgs)
            # fake_validity = self.discriminator(generated_imgs)

            # val_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            # self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=source_imgs.size(0))

            # 计算验证阶段的评价指标
            val_metrics = self.metric_manager_val.compute_metrics(generated_imgs, target_imgs, prefix="val")
            for name, value in val_metrics.items():
                self.log(name, value, on_step=False, on_epoch=True, prog_bar=True, batch_size=source_imgs.size(0))


    def test_step(self, batch, batch_idx):
        source_imgs = batch['A']
        target_imgs = batch['B']

        with torch.no_grad():
            generated_imgs = self(source_imgs)

            # 计算验证阶段的评价指标
            test_metrics = self.metric_manager_val.compute_metrics(generated_imgs, target_imgs, prefix="test")
            for name, value in test_metrics.items():
                self.log(name, value, on_step=False, on_epoch=True, prog_bar=True, batch_size=source_imgs.size(0))


    def configure_optimizers(self,  weight_decay_d=0.0):

        optimizer_type = self.optimizer_type

        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        if optimizer_type == 'adam':
            opt_g = torch.optim.Adam(self.parameters(), lr=lr, betas=(b1, b2))
        elif optimizer_type == 'rmsprop':
            opt_g = torch.optim.RMSprop(self.parameters(), lr=lr, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False)
        elif optimizer_type == 'sgd':
            opt_g = torch.optim.SGD(self.parameters(), lr=lr,nesterov = False)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # 动态调整学习率
        scheduler_g = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=10),
            'interval': 'epoch',
            'frequency': 1
        }

        return [opt_g], [scheduler_g]

        # return [opt_g, opt_d]
    def optimizer_step(self, 
            epoch: int, batch_idx: int, optimizer: Optimizer | LightningOptimizer, optimizer_closure: Callable[[], Any] | None = None) -> None:
        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def train_dataloader(self):
        data_loader = CustomDatasetDataLoader(mode='train')
        data_loader.initialize()
        return data_loader.load_data()  # 返回训练集的 DataLoader 对象

    def val_dataloader(self):
        val_data_loader = CustomDatasetDataLoader(mode='val')
        val_data_loader.initialize()
        return val_data_loader.load_data()  # 返回验证集的 DataLoader 对象

    def test_dataloader(self):
        val_data_loader = CustomDatasetDataLoader(mode='test')
        val_data_loader.initialize()
        return val_data_loader.load_data()  # 返回测试集的 DataLoader 对象


    # def on_epoch_end(self):
    #     # 获取验证数据加载器
    #     val_dataloader = self.val_dataloader()
    #     # 从验证集中获取一批数据（这里只取一个批次用于可视化）
    #     for batch in val_dataloader:
    #         source_imgs, target_imgs = batch['A'], batch['B']
    #         source_imgs = source_imgs.to(self.device)
    #         target_imgs = target_imgs.to(self.device)
    #         break  # 只取第一个批次
    #     # 使用生成器生成红外图像
    #     with torch.no_grad():  # 确保不计算梯度以节省内存
    #         generated_imgs = self(source_imgs)
    #     # 记录生成图像
    #     sample_imgs = generated_imgs[:3]
    #     grid = torchvision.utils.make_grid(tensor=sample_imgs, 
    #                                     nrow=3, normalize=True, value_range=(0,1), scale_each=True)
    #     self.logger.experiment.add_image('img_gen', grid, self.current_epoch) # 原为self.current_epoch
    #     # 记录真实图像
    #     targ_imgs = target_imgs[:3]
    #     grid = torchvision.utils.make_grid(tensor=targ_imgs, 
    #                                     nrow=4, normalize=True, value_range=(0,1), scale_each=True)
    #     self.logger.experiment.add_image('img_real', grid, self.current_epoch)

def main() -> None:

    # 1 INIT LIGHTNING MODEL
  
    model = GAN(n_critic=training_config['n_critic'])

    # 2 INIT LOGGER
    logger = TensorBoardLogger(save_dir="log_spvugan", name='AVIID1', version=0)

    # 3 INIT TRAINER WITH LOGGER
    trainer = Trainer(
                logger=logger,
                accelerator= 'cuda',
                devices='auto',
                max_epochs=training_config['epochs'], # 设置期望的训练轮数
                precision='bf16-mixed', # 使用混合精度
                check_val_every_n_epoch=1,  # 在每个 epoch 后验证一次
    )

    # 3 START TRAINING
    trainer.fit(model)

    # # 4 TEST THE MODEL ON TEST SET
    # trainer.test(model, dataloaders=model.test_dataloader(),
    #              ckpt_path='log_spvugan/AVIID1/version_0/checkpoints/epoch=99-step=6600.ckpt')  # 测试模型


if __name__ == '__main__':

    setup_system(seed=12) # 设置所有随机种子
    # torch.set_float32_matmul_precision('high')
    main()

'''
tensorboard --logdir tb_logs/
tensorboard --logdir media/dg/D/Py_Project/pytorch_lightning_spvugan/lightning_logs/

'''