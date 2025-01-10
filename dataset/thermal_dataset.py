import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from torchvision import transforms
from PIL import Image, ImageOps
import os
import numpy as np
from random import randint
import matplotlib.pyplot as plt

import yaml
import torch
from .base_dataset import BaseDataset
# from base_dataset import BaseDataset # 调试用



# 读取 YAML 配置文件
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 获取模型配置
model_config = config['model']
training_config = config['training']
data_config = config['data']
G_config = config['Generator']


## AVIID1
def make_thermal_dataset_aviid1(path=None, text_path=None, mode=None):

    assert os.path.isdir(text_path), '%s is not a valid directory' % text_path
    assert os.path.isdir(path), '%s is not a valid directory' % path
    images = []

    ## 判断数据集模式
    if mode == 'train':
        text_path = os.path.join(text_path, 'train_all.txt')
    elif mode == 'val':
        text_path = os.path.join(text_path, 'test_all.txt')
    elif mode == 'test':
        text_path = os.path.join(text_path, 'test_all.txt')
    assert os.path.isfile(text_path), '%s is not a valid file' % text_path

    with open(text_path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.split() # 按行读取，将每个图像的编号读出

        path_rgb = os.path.join(path, 'vis') # path/RGB
        path_rgb = os.path.join(path_rgb, 'v'+line[0]+'.png') # path/vis/vxxxxx.png

        path_ir = os.path.join(path, 'ir') # path/NIR
        path_ir = os.path.join(path_ir, 'i'+line[0]+'.png') # path/ir/ixxxxxx.png

        assert os.path.isfile(path_rgb), '%s is not a valid file' % path_rgb
        assert os.path.isfile(path_ir), '%s is not a valid file' % path_ir
        images.append({'A': path_rgb, 'B': path_ir, "annotation_file": os.path.join(path,
                                                                                    "..",
                                                                                    "annotations",
                                                                                    line[0]+'.txt')
                       })
    # np.random.seed(12)
    # np.random.shuffle(images)
    return images


## SMOD数据集读取代码
def make_thermal_dataset_smod(path=None, text_path=None, mode=None):

    assert os.path.isdir(text_path), '%s is not a valid directory' % text_path
    assert os.path.isdir(path), '%s is not a valid directory' % path
    images = []

    ## 判断数据集模式
    if mode == 'train':
        text_path = os.path.join(text_path, 'train_all.txt')
    elif mode == 'val':
        text_path = os.path.join(text_path, 'val_all.txt')
    elif mode == 'test':
        text_path = os.path.join(text_path, 'test_all.txt')
    assert os.path.isfile(text_path), '%s is not a valid file' % text_path

    with open(text_path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.split() # 按行读取，将每个图像的编号读出

        # path_rgb = os.path.join(path, 'VIS') # path/RGB
        path_rgb = os.path.join(path, line[0]+'_rgb.jpg') # path/xxxxxx_rgb.jpg

        # path_ir = os.path.join(path, 'NIR') # path/NIR
        path_ir = os.path.join(path, line[0]+'_tir.jpg') # path/xxxxxx_tir.jpg

        assert os.path.isfile(path_rgb), '%s is not a valid file' % path_rgb
        assert os.path.isfile(path_ir), '%s is not a valid file' % path_ir
        images.append({'A': path_rgb, 'B': path_ir, "annotation_file": os.path.join(path,
                                                                                    "..",
                                                                                    "annotations",
                                                                                    line[0]+'.txt')
                       })
    # np.random.seed(12)
    # np.random.shuffle(images)
    return images


## FLIR
def make_thermal_dataset_flir(path=None, text_path=None, mode=None):

    assert os.path.isdir(text_path), '%s is not a valid directory' % text_path
    assert os.path.isdir(path), '%s is not a valid directory' % path
    images = []

    ## 判断数据集模式
    if mode == 'train':
        text_path = os.path.join(text_path, 'train_all.txt')
    elif mode == 'val':
        text_path = os.path.join(text_path, 'val_all.txt')
    elif mode == 'test':
        text_path = os.path.join(text_path, 'test_all.txt')
    assert os.path.isfile(text_path), '%s is not a valid file' % text_path

    with open(text_path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.split() # 按行读取，将每个图像的编号读出

        path_rgb = os.path.join(path, 'RGB') # path/RGB
        path_rgb = os.path.join(path_rgb, line[0]+'.jpg') # path/RGB/FLIR_xxxxx.jpg

        path_ir = os.path.join(path, 'thermal_8_bit') # pat/thermal_8_bit
        path_ir = os.path.join(path_ir, line[0]+'.jpeg') # path/thermal_8_bit/irimg.jpeg

        assert os.path.isfile(path_rgb), '%s is not a valid file' % path_rgb
        assert os.path.isfile(path_ir), '%s is not a valid file' % path_ir
        images.append({'A': path_rgb, 'B': path_ir, "annotation_file": os.path.join(path,
                                                                                    "..",
                                                                                    "annotations",
                                                                                    line[0]+'.txt')
                       })
    # np.random.seed(12)
    # np.random.shuffle(images)
    return images


## KAIST
def make_thermal_dataset_kaist(path=None, text_path=None, mode=None):
    if path is None:
        path = '/cta/users/mehmet/rgbt-ped-detection/data/kaist-rgbt/images'
    if text_path is None:
        text_path = '/cta/users/mehmet/rgbt-ped-detection/data/scripts/imageSets/train-all-04.txt'
        text_path = '/cta/users/mehmet/rgbt-ped-detection/data/scripts/imageSets/test-all-20.txt'

    assert os.path.isdir(text_path), '%s is not a valid directory' % text_path
    assert os.path.isdir(path), '%s is not a valid directory' % path
    images = []

    ## 判断数据集模式
    if mode == 'train':
        text_path = os.path.join(text_path, 'day-06/train-day-s00s02-06.txt')
    elif mode == 'val':
        text_path = os.path.join(text_path, 'day-06/val-day-s07s08-06.txt')
    elif mode == 'test':
        text_path = os.path.join(text_path, 'day-06/test-day-s06-06.txt')
    assert os.path.isfile(text_path), '%s is not a valid file' % text_path

    with open(text_path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.split()[0] # 按空格读取
        line = line.split('/') # 按 / 读取
        path_set = os.path.join(path, line[0]) # path/setxx
        path_set_V = os.path.join(path_set, line[1]) # path/setxx/Vxxx

        path_ir = os.path.join(path_set_V, 'lwir') # path/setxx/Vxxx/lwir
        path_ir = os.path.join(path_ir, line[2]+'.jpg') # path/setxx/Vxxx/lwir/Ixxxxx.jpg

        path_rgb = os.path.join(path_set_V, 'visible') # path/setxx/Vxxx/visible
        path_rgb = os.path.join(path_rgb, line[2]+'.jpg') # path/setxx/Vxxx/visible/Ixxxxx.jpg

        assert os.path.isfile(path_rgb), '%s is not a valid file' % path_rgb
        assert os.path.isfile(path_ir), '%s is not a valid file' % path_ir
        images.append({'A': path_rgb, 'B': path_ir, "annotation_file": os.path.join(path,
                                                                                    "..",
                                                                                    "annotations",
                                                                                    line[0],
                                                                                    line[1],
                                                                                    line[2]+'.txt')
                       })

    # np.random.seed(12)
    # np.random.shuffle(images)
    return images

## VEDAI
def make_thermal_dataset_vedai(path=None, text_path=None, mode=None):

    assert os.path.isdir(text_path), '%s is not a valid directory' % text_path
    assert os.path.isdir(path), '%s is not a valid directory' % path
    images = []

    ## 判断数据集模式
    if mode == 'train':
        text_path = os.path.join(text_path, 'train_all.txt')
    elif mode == 'val':
        text_path = os.path.join(text_path, 'val_all.txt')
    elif mode == 'test':
        text_path = os.path.join(text_path, 'test_all.txt')
    assert os.path.isfile(text_path), '%s is not a valid file' % text_path

    with open(text_path) as f:
        lines = f.readlines()

        i = 0
    for line in lines:
        line = line.split() # 按行读取，将每个图像的编号读出

        path_rgb = os.path.join(path, 'VIS') # path/RGB
        path_rgb = os.path.join(path_rgb, line[0]+'_co.png') # path/VIS/xxxxxxxx_co.png

        path_ir = os.path.join(path, 'NIR') # path/NIR
        path_ir = os.path.join(path_ir, line[0]+'_ir.png') # path/NIR/xxxxxxxx_ir.png

        assert os.path.isfile(path_rgb), '%s is not a valid file' % path_rgb
        assert os.path.isfile(path_ir), '%s is not a valid file' % path_ir
        images.append({'A': path_rgb, 'B': path_ir, "annotation_file": os.path.join(path,
                                                                                    "..",
                                                                                    "annotations",
                                                                                    line[0]+'.txt')
                       })
    # np.random.seed(12)
    # np.random.shuffle(images)
 
        # print(images[i]) # 调试用，测试dataloader读取顺序
        # i = i + 1
    return images

################################################################################################################
################################################################################################################


class ThermalDataset(BaseDataset):
    def __init__(self, mode='train'):
        self.initialize(mode)

    def initialize(self, mode='train'):
        print('ThermalDataset')

        self.root = data_config['dataroot']  # 数据集图像文件路径
        self.dataset_mode = data_config['dataset_mode']  # 数据集模式
        self.dataset_files = data_config['dataset_files']  # 数据集路径文件路径（原text_path）
        self.resize_or_crop = data_config['resize_or_crop']  # 数据增强方法
        self.loadSize = data_config['load_size']  # 缩放图像至此大小
        self.fineSize = data_config['fine_size']  # 裁剪图像至此大小
        self.trans_direction = model_config['trans_direction']  # 图像转换方向
        self.input_nc = model_config['input_nc']  # 模型输入图像通道数
        self.output_nc = model_config['output_nc']  # 模型输出图像通道数
        self.mode = mode  # 训练、验证或测试模式

        self.dir_AB = os.path.join(self.root, mode)

        dataset_loader = {
            'VEDAI': make_thermal_dataset_vedai,
            'KAIST': make_thermal_dataset_kaist,
            'FLIR': make_thermal_dataset_flir,
            'SMOD': make_thermal_dataset_smod,
            'AVIID1': make_thermal_dataset_aviid1
        }

        if self.dataset_mode in dataset_loader:
            self.AB_paths = dataset_loader[self.dataset_mode](path=self.root, text_path=self.dataset_files, mode=mode)
        else:
            raise ValueError(f"Unknown dataset mode: {self.dataset_mode}")

        assert(self.resize_or_crop == 'resize_and_crop')

        ## 定义变换
        # 定义适用于RGB图像A的变换
        transform_list_A = [
            transforms.Resize(self.loadSize, Image.BICUBIC),
            transforms.ToTensor(),
        ]

        # 定义适用于灰度图像B的变换
        transform_list_B = [
            transforms.Resize(self.loadSize, Image.BICUBIC),
            transforms.Grayscale(num_output_channels=1),  # 确保是灰度图
            transforms.ToTensor(),
        ]

        if mode == 'train':
            transform_list_A.append(transforms.RandomCrop(self.fineSize))
            transform_list_A.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            transform_list_B.append(transforms.RandomCrop(self.fineSize))
            transform_list_B.append(transforms.Normalize([0.5], [0.5]))  # 单通道归一化
        else:
            transform_list_A.extend([
                transforms.CenterCrop(self.fineSize),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            transform_list_B.extend([
                transforms.CenterCrop(self.fineSize),
                transforms.Normalize([0.5], [0.5])  # 单通道归一化
            ])

        self.transform_A = transforms.Compose(transform_list_A)
        self.transform_B = transforms.Compose(transform_list_B)

    def __getitem__(self, index):
        A_path = self.AB_paths[index]['A']
        B_path = self.AB_paths[index]['B']

        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('L')  # 直接转换为灰度图

        A = self.transform_A(A)
        B = self.transform_B(B)  # 使用特定于灰度图的变换致

        if self.trans_direction == 'BtoA':
            input_nc = self.output_nc
            output_nc = self.input_nc
        else:
            input_nc = self.input_nc
            output_nc = self.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'ThermalDataset'
    
class CustomDatasetDataLoader(BaseDataset):
    def __init__(self, mode='train'):
        super().__init__()
        self.dataset = ThermalDataset(mode=mode)

        self.batchsize = training_config['batch_size']
        self.nThreads = training_config['nThreads']
        if mode == 'train':
            self.shuffle = data_config['shuffle']
        elif mode == 'val' or 'test':
            self.shuffle = False

    def name(self):
        return 'CustomDatasetDataLoader'
    
############### 改，添加 worker_init_fn用于多线程随机种子控制 ###############
    def _worker_init_fn(self, worker_id):
        """Initialize the numpy random seed for each worker."""
        base_seed = torch.initial_seed() % 2**32
        np.random.seed(base_seed + worker_id)
#######################################################################

    def initialize(self):
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batchsize,
            shuffle=self.shuffle,
            num_workers=int(self.nThreads),
            worker_init_fn=self._worker_init_fn,  # 添加 worker_init_fn用于多线程随机种子控制
            drop_last=True if self.shuffle else False  # 确保最后一个不完整的批次被丢弃（仅在训练时）
        )

    def load_data(self):
        return self.dataloader  # 返回 DataLoader 对象
    


if __name__ == '__main__':
    data_loader = CustomDatasetDataLoader(mode='train')
    data_loader.initialize()
    dataloader = data_loader.load_data()  # 返回训练集的 DataLoader 对象

    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1}")
        print("A:", batch['A'].size(), "B:", batch['B'].size())
        print("A_paths:", batch['A_paths'][0])
        print("B_paths:", batch['B_paths'][0])

        # 将PyTorch张量转换为NumPy数组，并转置维度以适应matplotlib
        img_A = batch['A'][0].numpy().transpose((1, 2, 0))
        img_B = batch['B'][0].numpy()[0]  # 因为B是单通道灰度图

        # 显示图像
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img_A)
        ax1.set_title('Image A')
        ax2.imshow(img_B, cmap='gray')
        ax2.set_title('Image B')
        plt.show()
        if i >= 1:  # 只显示前3个批次的数据用于调试
            break

