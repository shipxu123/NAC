import os
import json
import time
import pprint
import argparse
import datetime
from tqdm import tqdm
from easydict import EasyDict
from tensorboardX import SummaryWriter

import torch
from torch.autograd import Variable

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull, Reddit, PPI
from torch_geometric.utils import add_self_loops

from nac.utils.misc import makedir, create_logger, get_logger, AverageMeter, accuracy, load_state_model, load_state_optimizer,\
                         parse_config, set_seed, param_group_all, modify_state, save_load_split, gen_uniform_60_20_20_split, load_state_variable, gen_uniform_80_80_20_split
from nac.model import model_entry
from nac.optimizer import optim_entry
from nac.lr_scheduler import scheduler_entry
from nac.controller import controller_entry

from .base_solver import BaseSolver

class InductiveSolver(BaseSolver):

    def __init__(self, config_file):
        self.config_file = config_file
        self.config = parse_config(config_file)
        self.setup_env()
        self.build_model()
        self.build_optimizer()
        self.build_lr_scheduler()
        self.build_data()

    def setup_env(self):
        # directories
        self.path = EasyDict()
        self.path.root_path = os.path.dirname(self.config_file)
        self.path.save_path = os.path.join(self.path.root_path, 'checkpoints')
        self.path.event_path = os.path.join(self.path.root_path, 'events')
        self.path.result_path = os.path.join(self.path.root_path, 'results')
        makedir(self.path.save_path)
        makedir(self.path.event_path)
        makedir(self.path.result_path)

        self.tb_logger = SummaryWriter(self.path.event_path)

        # logger
        create_logger(os.path.join(self.path.root_path, 'log.txt'))
        self.logger = get_logger(__name__)
        self.logger.info(f'config: {pprint.pformat(self.config)}')

        # load pretrain checkpoint
        if hasattr(self.config.saver, 'pretrain'):
            self.state = torch.load(self.config.saver.pretrain.path, 'cpu')
            self.logger.info(f"Recovering from {self.config.saver.pretrain.path}, keys={list(self.state.keys())}")
            if hasattr(self.config.saver.pretrain, 'ignore'):
                self.state = modify_state(self.state, self.config.saver.pretrain.ignore)
            if 'last_iter' not in self.state:
                self.state['last_iter'] = 0
            if 'last_epoch' not in self.state:
                self.state['last_epoch'] = -1
        else:
            self.state = {}
            self.state['last_iter'] = 0
            self.state['last_epoch'] = -1

        # # others
        # torch.backends.cudnn.benchmark = True
        self.seed_base: int = int(self.config.seed_base)
        # set seed
        self.seed: int = self.seed_base
        set_seed(seed=self.seed)

    def build_model(self):
        self.model = model_entry(self.config.model)

        if 'model' in self.state:
            load_state_model(self.model, self.state['model'])
        else:
            load_state_model(self.model, self.state)


    def _build_optimizer(self, opt_config, model):
        # make param_groups
        pconfig = {}

        if opt_config.get('no_wd', False):
            pconfig['conv_b'] = {'weight_decay': 0.0}
            pconfig['linear_b'] = {'weight_decay': 0.0}
            pconfig['bn_w'] = {'weight_decay': 0.0}
            pconfig['bn_b'] = {'weight_decay': 0.0}

        if 'pconfig' in opt_config:
            pconfig.update(opt_config['pconfig'])

        param_group, type2num = param_group_all(model, pconfig)
        opt_config.kwargs.params = param_group
        return optim_entry(opt_config)

    def build_optimizer(self):
        self.optimizer = self._build_optimizer(self.config.optimizer, self.model)
        if 'optimizer' in self.state:
            load_state_optimizer(self.optimizer, self.state['optimizer'])

    def _build_lr_scheduler(self, lr_config, optimizer):
        lr_config.kwargs.optimizer = optimizer
        lr_config.kwargs.last_epoch = self.state['last_epoch']
        return scheduler_entry(lr_config)

    def build_lr_scheduler(self):
        self.lr_scheduler = self._build_lr_scheduler(self.config.lr_scheduler, self.optimizer)

    def build_data(self):
        """
        Specific for Iuductive tasks
        """
        if not getattr(self.config.data, 'max_epoch', False):
            self.config.data.max_epoch = self.config.lr_scheduler.kwargs.T_max

        if self.config.data.task == 'Amazon_Computers':
            dataset = Amazon(os.path.join(self.config.data.root, 'Amazon_Computers'), 'Computers')
        elif self.config.data.task == 'Coauthor_Physics':
            dataset = Coauthor(os.path.join(self.config.data.root, 'Coauthor_Physics'), 'Physics')
        elif self.config.data.task == 'Coauthor_CS':
            dataset = Coauthor(os.path.join(self.config.data.root, 'Coauthor_CS'), 'CS')
        elif self.config.data.task == 'Cora_Full':
            dataset = CoraFull(os.path.join(self.config.data.root, 'Cora_Full'))
        elif self.config.data.task == 'PubMed':
            dataset = Planetoid(self.config.data.root, 'PubMed')
        elif self.config.data.task == 'Cora':
            dataset = Planetoid(self.config.data.root, 'Cora')
        elif self.config.data.task == 'CiteSeer':
            dataset = Planetoid(self.config.data.root, 'CiteSeer')
        else:
            raise NotImplementedError(f'Dataset {self.config.data.task} is not supported!')

        data = dataset[0]
        data = save_load_split(data, gen_uniform_80_80_20_split)
        edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
        data.edge_index = edge_index

        self.data = {'loader': data}

    def _pre_train(self, model):
        self.meters = EasyDict()
        self.meters.batch_time = AverageMeter(self.config.saver.print_freq)
        self.meters.step_time = AverageMeter(self.config.saver.print_freq)
        self.meters.data_time = AverageMeter(self.config.saver.print_freq)
        self.meters.losses = AverageMeter(self.config.saver.print_freq)
        self.meters.top1 = AverageMeter(self.config.saver.print_freq)
        self.meters.top5 = AverageMeter(self.config.saver.print_freq)

        model.train()

        self.num_classes = self.config.model.kwargs.get('out_dim', 1000)
        self.topk = 5 if self.num_classes >= 5 else self.num_classes
        self.criterion = torch.nn.CrossEntropyLoss()

        self.mixup = self.config.get('mixup', 1.0)
        if self.mixup < 1.0:
            self.logger.info('using mixup with alpha of: {}'.format(self.mixup))

        # share same criterion with controller
        self.controller.set_criterion(self.criterion)

    def _train(self, model):
        self._pre_train(model=model)
        model.eval()

        iter_per_epoch = len(self.data['loader'])
        total_step = iter_per_epoch * self.config.data.max_epoch
        end = time.time()

        best_prec1_val , best_prec1_test = 0, 0

        for epoch in tqdm(range(0, self.config.data.max_epoch)):
            start_step = epoch * iter_per_epoch

            if start_step < self.state['last_iter']:
                continue

            self.lr_scheduler.step()
            # lr_scheduler.get_lr()[0] is the main lr
            current_lr = self.lr_scheduler.get_lr()[0]

            curr_step = start_step
            data = self.data['loader']

            # jumping over trained steps
            if curr_step < self.state['last_iter']:
                continue

            # architecture step for optizing alpha for one epoch
            self.controller.step(data, data, current_lr, self.optimizer)

            # measure elapsed time
            self.meters.batch_time.update(time.time() - end)

            # training logger
            if curr_step % 1 == 0:
                if getattr(model, 'arch_parameters', False):
                    self.tb_logger.add_histogram('na_alphas', model.na_alphas, curr_step)
                    self.tb_logger.add_histogram('sc_alphas', model.sc_alphas, curr_step)
                    self.tb_logger.add_histogram('la_alphas', model.la_alphas, curr_step)

                remain_secs = (total_step - curr_step) * self.meters.batch_time.avg
                remain_time = datetime.timedelta(seconds=round(remain_secs))
                finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
                log_msg = f'Iter: [{curr_step}/{total_step}]\t' \
                        f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                        f'Remaining Time {remain_time} ({finish_time})'
                self.logger.info(log_msg)

            end = time.time()

        # testing After training
        if curr_step >= 0 and (epoch + 1) % self.config.saver.val_epoch_freq == 0:
            metrics = self._validate(model=model)
            loss_val = metrics['loss']
            prec1_val = metrics['top1']

            metrics = self._evaluate(model=model)
            loss_test = metrics['loss']
            prec1_test = metrics['top1']

            # recording best accuracy performance based on validation accuracy
            if prec1_val > best_prec1_val:
                best_prec1_val = prec1_val
                best_prec1_test = prec1_test

            # testing logger
            self.tb_logger.add_scalar('loss_val', loss_val, curr_step)
            self.tb_logger.add_scalar('acc1_val', prec1_val, curr_step)
            self.tb_logger.add_scalar('loss_test', loss_test, curr_step)
            self.tb_logger.add_scalar('acc1_test', prec1_test, curr_step)

            # save ckpt
            if self.config.saver.save_many:
                ckpt_name = f'{self.path.save_path}/ckpt_{curr_step}.pth.tar'
            else:
                ckpt_name = f'{self.path.save_path}/ckpt.pth.tar'

            self.state['model'] = model.state_dict()
            self.state['optimizer'] = self.optimizer.state_dict()
            self.state['last_epoch'] = epoch
            self.state['last_iter'] = curr_step

        metrics = {}
        metrics['best_top1_val'] = best_prec1_val
        metrics['best_top1_test'] = best_prec1_test

        return metrics

    @torch.no_grad()
    def _evaluate(self, model):
        batch_time = AverageMeter(0)
        losses = AverageMeter(0)
        top1 = AverageMeter(0)
        top5 = AverageMeter(0)

        num_classes = self.config.model.kwargs.get('out_dim', 1000)
        topk = 5 if num_classes >= 5 else num_classes

        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        val_iter = len(self.data['loader'])
        end = time.time()

        data = self.data['loader']

        # get_data
        inp, target = data, Variable(data.y[data.test_mask])

        logits = model(inp)

        # measure f1_score/accuracy and record loss
        loss = criterion(logits[data.test_mask], target)
        prec1, prec5 = accuracy(logits[data.test_mask], target, topk=(1, topk))

        num = inp.size(0)
        losses.update(loss.item(), num)
        top1.update(prec1.item(), num)
        top5.update(prec5.item(), num)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        self.logger.info(f'Test: [{val_iter}/{val_iter}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})')

        # gather final results
        total_num = torch.Tensor([losses.count])
        loss_sum = torch.Tensor([losses.avg*losses.count])
        top1_sum = torch.Tensor([top1.avg*top1.count])
        top5_sum = torch.Tensor([top5.avg*top5.count])

        final_loss = loss_sum.item()/total_num.item()
        final_top1 = top1_sum.item()/total_num.item()
        final_top5 = top5_sum.item()/total_num.item()

        self.logger.info(f' * Prec@1 {final_top1:.3f}\t * Prec@5 {final_top5:.3f}\t\
            Loss {final_loss:.3f}\ttotal_num={total_num.item()}')

        model.train()
        metrics = {}
        metrics['loss'] = final_loss
        metrics['top1'] = final_top1
        metrics['top5'] = final_top5
        return metrics

    @torch.no_grad()
    def _validate(self, model):
        batch_time = AverageMeter(0)
        losses = AverageMeter(0)
        top1 = AverageMeter(0)
        top5 = AverageMeter(0)

        num_classes = self.config.model.kwargs.get('out_dim', 1000)
        topk = 5 if num_classes >= 5 else num_classes

        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        val_iter = len(self.data['loader'])
        end = time.time()

        data = self.data['loader']

        # get_data
        inp, target = data, Variable(data.y[data.val_mask])

        logits = model(inp)

        # measure f1_score/accuracy and record loss
        loss = criterion(logits[data.val_mask], target)
        prec1, prec5 = accuracy(logits[data.val_mask], target, topk=(1, topk))

        num = inp.size(0)
        losses.update(loss.item(), num)
        top1.update(prec1.item(), num)
        top5.update(prec5.item(), num)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        self.logger.info(f'Val: [{val_iter}/{val_iter}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})')

        # gather final results
        total_num = torch.Tensor([losses.count])
        loss_sum = torch.Tensor([losses.avg*losses.count])
        top1_sum = torch.Tensor([top1.avg*top1.count])
        top5_sum = torch.Tensor([top5.avg*top5.count])

        final_loss = loss_sum.item()/total_num.item()
        final_top1 = top1_sum.item()/total_num.item()
        final_top5 = top5_sum.item()/total_num.item()

        self.logger.info(f' * Prec@1 {final_top1:.3f}\t * Prec@5 {final_top5:.3f}\t\
            Loss {final_loss:.3f}\ttotal_num={total_num.item()}')

        model.train()
        metrics = {}
        metrics['loss'] = final_loss
        metrics['top1'] = final_top1
        metrics['top5'] = final_top5
        return metrics

    def train(self):
        self._train(model=self.model)

    def evaluate(self):
        self._evaluate(model=self.model)

def main():
    parser = argparse.ArgumentParser(description='Graph Neural archtecture search Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--phase', default='train_search')

    args = parser.parse_args()
    # build solver
    solver = InductiveSolver(args.config)

    # evaluate or fintune or train_search
    if args.phase == 'train':
        if solver.state['last_epoch'] <= solver.config.data.max_epoch:
            solver.train()
        else:
            solver.logger.info('Training has been completed to max_epoch!')
    elif args.phase == 'evaluate':
        solver.evaluate()
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()