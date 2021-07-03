import torch
import time
import os
import numpy as np

__dirpath__ = os.path.dirname(__file__)
__version__ = '1.1.0'

if torch.cuda.is_available():
    default_device = torch.device('cuda')
else:
    default_device = torch.device('cpu')

backend = torch
P = torch
from torch import no_grad
# from . import visualise

def nograd(f):
    with no_grad():
        return f

def Tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, list) or isinstance(x, tuple):
        return torch.Tensor(x)
    else:
        return torch.Tensor([x])

def fromfile(fname, dtype=np.float32):
    return torch.from_numpy(np.fromfile(fname, dtype=dtype))

def Array(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    elif isinstance(x, list) or isinstance(x, tuple):
        return np.array(x)
    else:
        return np.array(x)

class block(object):

    def __init__(self, name, timeit=False):
        if timeit:
            self.time_start = time.time()
            self.name = name
        self.timeit = timeit

    def __enter__(self, name=None):
        return self

    def __exit__(self, a, b, c):
        if self.timeit:
            print(f'block[{self.name}] cost time: {time.time() - self.time_start}')

def read_raw10(path, shape=(800,1280)):
    img = np.fromfile(path, dtype=np.uint16)
    img = np.reshape(img, shape)
    img = np.rot90(img, 1)
    convert_img = cv2.convertScaleAbs(img, alpha=1. / (pow(2, 2)))

    return convert_img

#  def CreateDirForFile(filename):
#      dir_name = os.path.split(filename)[0]
#      im.checkfiledir

def ChangeSpaceOfFiles(root, src_word=' ', dst_word='_', change_all=True):
    import shutil
    import os
    for dir, _, names in os.walk(root):
        for name in names:
            if src_word in name:
                src_filename = os.path.join(dir, name)
                if change_all:
                    dst_filename = os.path.join(dir, name.replace(src_word, dst_word))
                else:
                    dst_filename = os.path.join(dir, name.replace(src_word, dst_word, 1))


                shutil.move(src_filename, dst_filename)


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from . import utils
from . import DataLoader
from . import Trainer

#  F = functional

from .math import *
from .usual_command import *


#  from .Random import *
#  from . import visualise
#  from .kill_process import *
#  from .LogManager import LogManager
#  from . import Trainer
#  from . import DataLoader
#
#  from . import timer
