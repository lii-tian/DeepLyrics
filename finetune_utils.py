import torch
import transformers
import numpy as np
import copy
import logging
import torch.nn.functional as F
import math
import os
from dataclasses import dataclass, field
from glob import glob
from typing import Optional
import torch.nn as nn
from torch.utils.data import ConcatDataset


class lowrank(nn.Module):
    def __init__(self, conv1dmodule: nn.Module, rank: int):
        super().__init__()
        self.base_module = conv1dmodule
        d1 = self.base_module.weight.shape[0]
        d2 = self.base_module.weight.shape[1]
        #initialize A as 0 tensor, randomly initialize B
        self.A = nn.Parameter(torch.zeros(d1, rank))
        self.B = nn.Parameter(torch.randn(d2, rank))

    def forward(self, x):
        #out = original + A @ B.T
        out = self.base_module(x) + (x @ self.A)@self.B.T
        return out

def parameters_to_fine_tune(model,layer):
    if layer == 'all':
        return model.parameters()
    elif layer == 'last':#only finetune the first transformer
        return list(model.transformer.h[-1].parameters())
    elif layer == 'first':#only finetune the last transformer
        return list(model.transformer.h[0].parameters())
    elif layer == 'middle':# only finetune a middle layer
        start = len(model.transformer.h)//2 - 1
        return list(model.transformer.h[start:start+2].parameters())
    elif layer = 'prefix':
        return []
    elif layer == 'lowrank':  #finetune the low rank adaptation of the attention modules A and B
        params = []
        for m in model.transformer.h:
            params.append(m.attn.c_attn.A)
            params.append(m.attn.c_attn.B)
            params.append(m.attn.c_attn.A)
            params.append(m.attn.c_attn.B)
            params.append(m.mlp.c_fc.A)
            params.append(m.mlp.c_fc.B)
            params.append(m.mlp.c_proj.A)
            params.append(m.mlp.c_proj.B)
        return params
