import torch
import numpy as np
import random


def make_deterministic(seed: int):
    """sets the seed to the random modules of python, numpy and pytorch

    Args:
        seed (int): the seed to use for all 3 libraries
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('seed:', seed)
