import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import matplotlib
# matplotlib.use('Agg')
# %matplotlib notebook

import os
import datetime as dt
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

use_cuda = torch.cuda.is_available()
# use_cuda = False
print("Is CUDA available? ", use_cuda)