import os.path
import torchvision.transforms as transforms
import numpy as np
from .base_dataset import BaseDataset
# from data.image_folder import make_dataset
# from data.image_folder import make_thermal_dataset, is_image_file
from PIL import Image, ImageOps

import yaml
import torch



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

        self.root = data_config['dataroot'] # 数据集图像文件路径
        self.dataset_mode = data_config['dataset_mode'] # 数据集模式
        self.dataset_files = data_config['dataset_files'] # 数据集路径文件路径（原text_path）
        self.resize_or_crop = data_config['resize_or_crop'] # 数据增强方法
        self.loadSize = data_config['load_size'] # 缩放图像至此大小
        self.fineSize = data_config['fine_size'] # 裁剪图像至此大小
        self.trans_direction = model_config['trans_direction'] # 图像转换方向
        self.input_nc = model_config['input_nc'] # 模型输入图像通道数
        self.output_nc = model_config['output_nc'] # 模型输入图像通道数

        self.dir_AB = os.path.join(self.root, mode)

        if self.dataset_mode =='VEDAI':
            self.AB_paths = make_thermal_dataset_vedai(path=self.root, text_path=self.dataset_files, mode=mode)
        elif self.dataset_mode == 'KAIST':
            self.AB_paths = make_thermal_dataset_kaist(path=self.root, text_path=self.dataset_files, mode=mode)
        elif self.dataset_mode == 'FLIR':
            self.AB_paths = make_thermal_dataset_flir(path=self.root, text_path=self.dataset_files, mode=mode)
        elif self.dataset_mode == 'SMOD':
            self.AB_paths = make_thermal_dataset_smod(path=self.root, text_path=self.dataset_files, mode=mode)
        elif self.dataset_mode == 'AVIID1':
            self.AB_paths = make_thermal_dataset_aviid1(path=self.root, text_path=self.dataset_files, mode=mode)
        assert(self.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        # AB_path = self.AB_paths[index]
        A_path = self.AB_paths[index]['A']
        B_path = self.AB_paths[index]['B']

        ## 改，改为不读取annotation_file
        # ann_path = self.AB_paths[index]['annotation_file']

        
        A = Image.open(A_path).convert('RGB')
        #A = A.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)

        ################## 改，若使用AVIID1，调整大小至512 #################
        if self.dataset_mode  == 'AVIID1':
            A = A.resize((self.loadSize, self.loadSize), Image.BICUBIC)
        ################################################################

        A = transforms.ToTensor()(A.copy())
        
        B = Image.open(B_path)
        B = ImageOps.grayscale(B)
        #  B = B.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)

        ################# 改，若使用AVIID1，调整大小至512 ##################
        if self.dataset_mode  == 'AVIID1':
            B = B.resize((self.loadSize, self.loadSize), Image.BICUBIC)
        ################################################################

        B = transforms.ToTensor()(B.copy()).float()


        w_total = A.size(2)
        w = int(w_total)
        h = A.size(1)
        w_offset = max(0, (w - self.fineSize - 1)//2)  # random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = max(0, (h - self.fineSize - 1)//2)  # random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.fineSize,
               w_offset:w_offset + self.fineSize]
        B = B[:, h_offset:h_offset + self.fineSize,
               w_offset:w_offset + self.fineSize]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize([0.5], [0.5])(B)

        if self.trans_direction == 'BtoA':
            input_nc = self.output_nc
            output_nc = self.input_nc
        else:
            input_nc = self.input_nc
            output_nc = self.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path, 
                # "annotation_file": ann_path
                } # 改，改为不读取annotation_file

        # print({'A_paths': A_path, 'B_paths': B_path})


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

        # np.random.seed(12)
    def initialize(self):
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batchsize,
            shuffle=self.shuffle,
            num_workers=int(self.nThreads),
            worker_init_fn=self._worker_init_fn) # 改，添加 worker_init_fn用于多线程随机种子控制


    def load_data(self):
        return self.dataloader  # 返回 DataLoader 对象

    # def __len__(self):
    #     return min(len(self.dataset), self.opt.max_dataset_size)

    # def __iter__(self):
    #     for i, data in enumerate(self.dataloader):
    #         if i >= self.opt.max_dataset_size:
    #             break
    #         yield data

    # def __len__(self):
    #     return len(self.dataset)

    # def __iter__(self):
    #     for data in enumerate(self.dataloader):
    #         yield data



if __name__ == '__main__':

    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize()
