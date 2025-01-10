import copy
import torch
from torch import nn

def extract_name_kwargs(obj):
    '''
    从对象中提取名称和关键字参数。
    主要用于处理可能包含名称和参数的输入对象，以便在后续代码中更方便地使用这些信息。
    '''
    if isinstance(obj, dict): # 若对象为字典类型
        obj    = copy.copy(obj)  #创建一个 obj 的浅拷贝，以避免修改原始字典 
        name   = obj.pop('name') # 提取字典中的 name 键，并将其值赋给 name 变量。同时这个键值对会从字典中删除
        kwargs = obj             # 将剩余的字典内容赋给 kwargs 变量
    else: # 若对象非字典类型
        name   = obj # 直接将 obj 赋给 name 变量
        kwargs = {}  # kwargs 设置为空字典

    return (name, kwargs) # 返回一个包含 name 和 kwargs 的元组

def get_norm_layer(norm, features):
    '''
    获取归一化层
    输入：归一化层类型，特征数量
    '''
    ## 从 norm 参数中提取名称和关键字参数：规范化层的名称，额外参数
    name, kwargs = extract_name_kwargs(norm)

    if name is None: # 若name为none，则不做任何操作
        return nn.Identity(**kwargs)

    if name == 'layer': # 若name为layer，则使用层归一化
        return nn.LayerNorm((features,), **kwargs)

    if name == 'batch': # 若name为batch，则使用批归一化
        return nn.BatchNorm2d(features, **kwargs)

    if name == 'instance': # 若name为instance，则使用实例归一化
        return nn.InstanceNorm2d(features, **kwargs)

    raise ValueError("Unknown Layer: '%s'" % name) # name为其他则报错

def get_norm_layer_fn(norm):
    '''
    返回一个用于获取规范化层的函数
    '''
    ## 返回一个匿名函数（lambda 函数），它接受一个参数 features，并调用 get_norm_layer 函数，将 norm 和 features 传递给它。
    return lambda features : get_norm_layer(norm, features)

def get_activ_layer(activ):
    '''
    选择并获取激活函数
    '''
    ## 从 activ 参数中提取名称和关键字参数：激活函数的名称
    name, kwargs = extract_name_kwargs(activ)

    if (name is None) or (name == 'linear'): # 若name为none或linear，则不做任何操作
        return nn.Identity()

    if name == 'gelu':
        return nn.GELU(**kwargs)

    if name == 'relu': 
        return nn.ReLU(**kwargs)

    if name == 'leakyrelu':
        return nn.LeakyReLU(**kwargs)

    if name == 'tanh':
        return nn.Tanh()

    if name == 'sigmoid':
        return nn.Sigmoid()

    raise ValueError("Unknown activation: '%s'" % name)

def select_optimizer(parameters, optimizer):
    '''
    选择优化器
    '''
    ## 从 optimizer 参数中提取名称和关键字参数：优化器的名称
    name, kwargs = extract_name_kwargs(optimizer) 

    if name == 'AdamW':
        return torch.optim.AdamW(parameters, **kwargs)

    if name == 'Adam':
        return torch.optim.Adam(parameters, **kwargs)

    raise ValueError("Unknown optimizer: '%s'" % name)

def select_loss(loss):
    '''
    选择loss（l1，l2）
    '''
    name, kwargs = extract_name_kwargs(loss)

    if name.lower() in [ 'l1', 'mae' ]:
        return nn.L1Loss(**kwargs)

    if name.lower() in [ 'l2', 'mse' ]:
        return nn.MSELoss(**kwargs)

    raise ValueError("Unknown loss: '%s'" % name)

