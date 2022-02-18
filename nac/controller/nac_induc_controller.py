import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pdb

class NACInductiveController(object):

    def __init__(self, config):
        self.config = config
        self.build_sane_settings()

    def build_sane_settings(self):
        self.network_momentum = self.config.momentum
        self.network_weight_decay = self.config.weight_decay
        self.unrolled = self.config.unrolled

        # subnet
        self.subnet = self.config.get('subnet', None)

    # 在solver中进行model的初始化
    def set_supernet(self, model):
        self.model = model

    # 在solver中进行logger的初始化
    def set_logger(self, logger):
        self.logger = logger

    # 在solver中进行目标函数的初始化
    def set_criterion(self, criterion):
        self.criterion = criterion

    # 依赖于model的初始化
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

        if getattr(self.config, 'sparse', False):
            p = getattr(self.config.sparse, 'norm', 2)
            _lambda = getattr(self.config.sparse, 'lambda', 0.001)
            self.logger.info(f'original loss = {loss}')
            if getattr(self.config.sparse, 'na_sparse', True):
                na_reg_loss = _lambda * torch.norm(self.model.na_weights, p=p)
                self.logger.info(f'na sparse loss = {na_reg_loss}')
                loss += na_reg_loss
            if getattr(self.config.sparse, 'sc_sparse', True):
                sc_reg_loss = _lambda * torch.norm(self.model.sc_weights, p=p)
                self.logger.info(f'sc sparse loss = {sc_reg_loss}')
                loss += sc_reg_loss
            if getattr(self.config.sparse, 'la_sparse', True):
                la_reg_loss = _lambda * torch.norm(self.model.la_weights, p=p)
                self.logger.info(f'la sparse loss = {la_reg_loss}')
                loss += la_reg_loss
        loss.backward()

    def build_active_subnet(self, subnet_settings):
        return self.model.build_active_subnet(subnet_settings)