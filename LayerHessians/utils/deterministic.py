#!/usr/bin/env python3

import torch
import numpy
def make_deterministic(cuda):

    torch.manual_seed(0)
    numpy.random.seed(0)
    if cuda is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False    
