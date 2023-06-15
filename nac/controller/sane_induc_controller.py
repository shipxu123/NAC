import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pdb

class SANEInductiveController(object):

    def __init__(self, config):
        self.config = config
        self.build_sane_settings()

    def build_sane_settings(self):
        self.network_momentum = self.config.momentum
        self.network_weight_decay = self.config.weight_decay
        self.unrolled = self.config.unrolled

        # subnet
        self.subnet = self.config.get('subnet', None)

    # init model in solver
    def set_supernet(self, model):
        self.model = model

    # init logger in solver
    def set_logger(self, logger):
        self.logger = logger

    # init criterion in solver
    def set_criterion(self, criterion):
        self.criterion = criterion

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
            lr=self.config.arch_learning_rate, betas=(0.5, 0.999), weight_decay=self.config.arch_weight_decay)

    def step(self, data):
        if self.subnet != None:
            return

        self.optimizer.zero_grad()
        self._backward_step(data)
        self.optimizer.step()

    def _backward_step(self, data):
        inp, target = data, Variable(data.y[data.val_mask], requires_grad=False)
        logit = self.model(inp)
        loss = self.criterion(logit[data.val_mask], target)
        loss.backward()

    def build_active_subnet(self, subnet_settings):
        return self.model.build_active_subnet(subnet_settings)