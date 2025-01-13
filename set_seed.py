import numpy as np
import random
import torch
import os


def setup_system(seed, cudnn_benchmark=False, cudnn_deterministic=True)->None:
    ''' Set seeds for reproductable training '''


    # python
    random.seed(seed)

    # numpy
    np.random.seed(seed)

    # pytorch
    torch.manual_seed(seed) # 设置 CPU 随机种子
    torch.cuda.manual_seed_all(seed) # 设置每个 GPU 的随机种子
    if torch.cuda.is_available():
        ## 若为 True，cuDNN 会在开始时花费额外的时间来选择最适合当前硬件的卷积实现。对于训练过程较长的任务是有利，因为可以提高卷积计算的速度。
        torch.backends.cudnn.benchmark = cudnn_benchmark
        ## 若为 True，cuDNN 将使用确定性的卷积算法。可能会降低性能，但可保证实验结果完全可复现。
        torch.backends.cudnn.deterministic = cudnn_deterministic
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)
    print('seed = ', seed)

