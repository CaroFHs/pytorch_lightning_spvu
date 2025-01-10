import os
from pytorch_lightning.core.optimizer import LightningOptimizer
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import torchvision
import torch.nn.init as init

from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from torch.amp import GradScaler, autocast

import yaml
from TransUNet import TransUNet_G, DiscriminatorManager
from dataset.thermal_dataset import CustomDatasetDataLoader
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
        # 在所有子模块被实例化后，调用初始化方法
        # self.init_weights(init_type='normal')

    def init_weights(self, init_type='kaiming', gain=0.02):
        '''
        初始化模型的权重。
        :param init_type: 初始化类型 ('normal', 'xavier', 'kaiming', 'orthogonal')
        :param gain: 放大因子，用于调整初始化权重的标准差或尺度。
        '''
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
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

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
            ngf = G_config['input_layer_output'],  # 第一层卷积输入通道数
            use_dropout = G_config['unet_use_dropout'], # unet是否使用dropout
            resolution = model_config['output_size'],  # 输入分辨率
            )

    def forward(self, x):
        y = self.netD(x)
        return y


class WGANGP(LightningModule):

    def __init__(self,
                n_critic=5,  # 每n_critic次判别器训练后训练一次生成器
                lr=training_config['lr'],
                b1=training_config['b1'],
                b2=training_config['b2'],
                batch_size=training_config['batch_size'],
                image_shape=G_config['image_shape'],  # 输入图像的形状
                **kwargs):
        super().__init__()

        self.automatic_optimization = False  # 禁用自动优化
        self.save_hyperparameters()
    
        self.n_critic = n_critic
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.lambda_gp = training_config['lambda_gp']
        # networks
        self.generator = Generator(
            which_model_netG=G_config['which_model_netG'],
            image_shape=self.image_shape
        )
        self.discriminator = Discriminator(
            which_model_netD=D_config['which_model_netD']
        )
        initialize_weights(self.generator, init_type='kaiming')
        initialize_weights(self.discriminator, init_type='kaiming')

        # print(self.discriminator)

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

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""

        # 真实样本和假样本之间插值的随机权重项
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

        # 使用新图计算插值样本上D的输出
        d_interpolates = self.discriminator(interpolates)

        # 创建一个与d_interpolates形状匹配的张量用于grad_outputs
        fake = torch.ones(fake_samples.shape[0], 1).to(self.device)

        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True, # 每次迭代保留计算图
            only_inputs=True,
        )[0]

        # 检查是否有 NaN 或者无穷大的梯度
        if torch.isnan(gradients).any() or torch.isinf(gradients).any():
            raise RuntimeError("Detected NaN or Inf in gradients.")
        
        gradients = gradients.view(gradients.size(0), -1) # gradients 张量展平为二维

        print(gradients.max(), gradients.min())
        # self.log('g_mean', gradients.mean(), on_step=False, on_epoch=True, prog_bar=True, batch_size=8)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp # 整个批次的平均梯度惩罚

        return gradient_penalty
    
    # def optimizer_step(self, epoch: int, batch_idx: int, optimizer: Optimizer | LightningOptimizer, optimizer_closure: Callable[[], Any] | None = None) -> None:

    #     return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def training_step(self, batch, batch_idx):

        # 读取图像
        source_imgs = batch['A']
        target_imgs = batch['B']
        if torch.isnan(source_imgs).any() or torch.isinf(source_imgs).any():
            raise ValueError("Source_imgs input contains NaN or Inf values")
        if torch.isnan(target_imgs).any() or torch.isinf(target_imgs).any():
            raise ValueError("Target_imgs nput contains NaN or Inf values")
        
        # 获取优化器
        opt_g, opt_d = self.optimizers()

####

        if batch_idx % self.n_critic == 0:
            generated_imgs = self(source_imgs)
            if torch.isnan(generated_imgs).any() or torch.isinf(generated_imgs).any():
                raise ValueError("Detected NaN or Inf in generated images.")
            
            # 记录生成图像
            sample_imgs = generated_imgs[:4]
            grid = torchvision.utils.make_grid(tensor=sample_imgs, 
                                            nrow=4, normalize=True, value_range=(0,1), scale_each=True)
            self.logger.experiment.add_image('image_gen', grid, self.current_epoch) # 原为self.current_epoch
            # 记录真实图像
            targ_imgs = target_imgs[:4]
            grid = torchvision.utils.make_grid(tensor=targ_imgs, 
                                            nrow=4, normalize=True, value_range=(0,1), scale_each=True)
            self.logger.experiment.add_image('img_real', grid, self.current_epoch)

            # 优化生成器
            opt_g.zero_grad()
            # self.manual_backward(total_g_loss)

            ## 生成器损失
            with autocast(device_type='cuda'):  # 使计算在混合精度下进行
                Ggan_loss = -torch.mean(self.discriminator(generated_imgs)) # 生成器gan损失
                gen_losses = self.loss_manager.compute_losses(generated_imgs, target_imgs) # 生成损失
                total_g_loss = Ggan_loss + gen_losses['total_loss'] # 总损失

            # 使用混合精度进行反向传播，先缩放损失
            scaled_g_loss = self.scaler.scale(total_g_loss)  # 使用混合精度进行反向传播，先缩放损失
            # scaled_g_loss.backward()  # 使用Scaler的backward来避免梯度问题

            self.manual_backward(scaled_g_loss)

            self.scaler.unscale_(opt_g) # 解缩放
            skip_step = False

            # 立即检查梯度中的 NaN 或 Inf
            for name, param in self.generator.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"NaN or Inf found in {name}'s gradient!")
                    print(f"Gradient: {param.grad}")
                    # raise ValueError(f"Detected NaN or Inf in {name}'s gradient")
                    skip_step = True
                    break

            # 若梯度异常，则跳过此次优化
            if not skip_step:
                # torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=3.0)
                torch.nn.utils.clip_grad_value_(self.generator.parameters(), clip_value=1.0)  # 梯度裁剪
                self.scaler.step(opt_g)
                self.scaler.update()

            # 记录生成器阶段损失
            self.log("g_loss", Ggan_loss, prog_bar=True, batch_size=source_imgs.size(0))
            for name, value in gen_losses.items():
                self.log(name, value, prog_bar=True, batch_size=source_imgs.size(0))

            # 计算训练阶段的评价指标
            train_metrics = self.metric_manager_train.compute_metrics(generated_imgs, target_imgs, prefix="train")
            for name, value in train_metrics.items():
                self.log(name, value, on_step=False, on_epoch=True, prog_bar=True, batch_size=source_imgs.size(0))

            return {'loss': total_g_loss}
        
        else:
            # 判别器优化
            with torch.no_grad():
                generated_imgs = self(source_imgs).detach()

            with autocast(device_type='cuda', enabled=False):  # 强制使用 FP32
                real_validity = self.discriminator(target_imgs.to(dtype=torch.float32))
                fake_validity = self.discriminator(generated_imgs.detach().to(dtype=torch.float32))
                gradient_penalty = self.compute_gradient_penalty(
                    target_imgs.to(dtype=torch.float32), 
                    generated_imgs.to(dtype=torch.float32)
                )

            r_val = torch.mean(real_validity).item()
            f_val = torch.mean(fake_validity).item()
            self.log("r_val", r_val, prog_bar=True, batch_size=source_imgs.size(0))
            self.log("f_val", f_val, prog_bar=True, batch_size=source_imgs.size(0))
            self.log("gp", gradient_penalty, prog_bar=True, batch_size=source_imgs.size(0))


            # if abs(r_val + f_val) > 0.001:  # 设置margin避免判别器区分能力过强
            #     self.n_critic = 5
            # else:
            #     self.n_critic = 2
            self.log("n_critic", self.n_critic, prog_bar=True, batch_size=source_imgs.size(0))

            # 计算判别器损失
            with autocast(device_type='cuda'):  # 在混合精度下
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty

            scaled_d_loss = self.scaler.scale(d_loss)  # 缩放损失
            self.manual_backward(scaled_d_loss)  # 反向传播

            # 梯度裁剪和检查
            self.scaler.unscale_(opt_d) # 解缩放
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=3.0) # 梯度裁剪

            # 立即检查梯度中的 NaN 或 Inf
            for name, param in self.generator.named_parameters():  # 注意这里应该是 generator 的参数
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"NaN or Inf found in {name}'s gradient!")
                    print(f"Gradient: {param.grad}")
                    # raise ValueError(f"Detected NaN or Inf in {name}'s gradient")

            # 优化步骤
            self.scaler.step(opt_d)
            self.scaler.update()

            self.log("d_loss", d_loss, prog_bar=True, batch_size=source_imgs.size(0))
            return d_loss


    def validation_step(self, batch, batch_idx):
        source_imgs = batch['A']
        target_imgs = batch['B']

        with torch.no_grad():
            generated_imgs = self(source_imgs)
            real_validity = self.discriminator(target_imgs)
            fake_validity = self.discriminator(generated_imgs)

            val_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=source_imgs.size(0))

            # 计算验证阶段的评价指标
            val_metrics = self.metric_manager_val.compute_metrics(generated_imgs, target_imgs, prefix="val")
            for name, value in val_metrics.items():
                self.log(name, value, on_step=False, on_epoch=True, prog_bar=True, batch_size=source_imgs.size(0))


    def test_step(self, batch, batch_idx):
        source_imgs = batch['A']
        target_imgs = batch['B']

        with torch.no_grad():
            generated_imgs = self(source_imgs)
            real_validity = self.discriminator(target_imgs)
            fake_validity = self.discriminator(generated_imgs)

            test_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=source_imgs.size(0))

            # 计算验证阶段的评价指标
            val_metrics = self.metric_manager_val.compute_metrics(generated_imgs, target_imgs, prefix="val")
            for name, value in val_metrics.items():
                self.log(name, value, on_step=False, on_epoch=True, prog_bar=True, batch_size=source_imgs.size(0))


    def configure_optimizers(self,  weight_decay_d=0.0):

        optimizer_type = self.optimizer_type

        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        if optimizer_type == 'adam':
            opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
            opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay_d)
        elif optimizer_type == 'rmsprop':
            opt_g = torch.optim.RMSprop(self.generator.parameters(), lr=lr, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False)
            opt_d = torch.optim.RMSprop(self.discriminator.parameters(), lr=lr, alpha=0.99, eps=1e-8, weight_decay=weight_decay_d, momentum=0, centered=False)
        elif optimizer_type == 'sgd':
            opt_g = torch.optim.SGD(params=self.generator.parameters(), lr=lr,nesterov = False)
            opt_d = torch.optim.SGD(params=self.discriminator.parameters(), lr=lr,nesterov = False)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # 动态调整学习率
        scheduler_g = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=10),
            'interval': 'epoch',
            'frequency': 1
        }
        scheduler_d = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=10),
            'interval': 'epoch',
            'frequency': 1
        }

        return [opt_d, opt_g], [scheduler_d, scheduler_g]

        # return [opt_g, opt_d]


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

    def on_epoch_end(self):
        z = torch.randn(6, *self.hparams.image_shape[1:]).to(self.device)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)


def main() -> None:

    # 1 INIT LIGHTNING MODEL
  
    model = WGANGP(n_critic=training_config['n_critic'])

    # 2 INIT TRAINER
    trainer = Trainer(
                # callbacks=[RichProgressBar()],
                accelerator= 'cuda',
                devices='auto',
                max_epochs=training_config['epochs'], # 设置期望的训练轮数
                precision='bf16-mixed', # 使用混合精度
                check_val_every_n_epoch=1,  # 在每个 epoch 后验证一次
    )

    # 3 START TRAINING
    trainer.fit(model)

    # 4 TEST THE MODEL ON TEST SET
    # trainer.test(model, dataloaders=model.test_dataloader())  # 测试模型


if __name__ == '__main__':

    setup_system(seed=12) # 设置所有随机种子
    torch.set_float32_matmul_precision('high')
    main()

'''
tensorboard --logdir tb_logs/
tensorboard --logdir media/dg/D/Py_Project/pytorch_lightning_spvugan/lightning_logs/

'''