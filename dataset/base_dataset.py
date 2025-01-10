import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms


import yaml


# 读取 YAML 配置文件
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 获取模型配置
data_config = config['data']
loadSize = data_config['load_size']
fineSize = data_config['fine_size']
resize_or_crop = data_config['resize_or_crop']

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'


def get_transform(isTrain, no_flip):
    transform_list = []
    if resize_or_crop == 'resize_and_crop':
        osize = [loadSize, loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(fineSize))
    elif resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(fineSize))
    elif resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, fineSize)))
    elif resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, loadSize)))
        transform_list.append(transforms.RandomCrop(fineSize))

    if isTrain and not no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
