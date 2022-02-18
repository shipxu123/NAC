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

from nac.utils.misc import accuracy, load_state_variable, save_load_split, gen_uniform_60_20_20_split

from nac.controller import controller_entry
from .induc_solver import InductiveSolver

class NACInductiveSolver(InductiveSolver):

    def __init__(self, config_file):
        super(NACInductiveSolver, self).__init__(config_file)

        # set up NAS controller
        self.controller = controller_entry(self.config.nas)
        self.controller.set_supernet(self.model)
        self.controller.set_logger(self.logger)
        self.controller.init_optimizer()

    def build_model(self):
        super(NACInductiveSolver, self).build_model()

        if getattr(self.model, 'arch_parameters', False) and 'arch_parameters' in self.state:
            arch_parameters = self.model.arch_parameters()
            for i, state in enumerate(self.state['arch_parameters']):
                load_state_variable(arch_parameters[i], state)

    def updata_weight_step(self, model, data):
        start_time = time.time()

        # get_data
        inp, target = data, Variable(data.y[data.train_mask], requires_grad=False)

        # measure data loading time
        self.meters.data_time.update(time.time() - start_time)

        # forward
        logits = model(inp)

        # clear gradient
        self.optimizer.zero_grad()

        # compute and update gradient
        loss = self.criterion(logits[data.train_mask], target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits[data.train_mask], target, topk=(1, self.topk))

        reduced_loss = loss.clone()
        reduced_prec1 = prec1
        reduced_prec5 = prec5

        self.meters.losses.reduce_update(reduced_loss)
        self.meters.top1.reduce_update(reduced_prec1)
        self.meters.top5.reduce_update(reduced_prec5)

        # compute and update gradient
        loss.backward()

        # # Clip Grad norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)

        # # compute and update gradient
        self.optimizer.step()

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

            # architecture step: optizing alpha for one epoch
            self.controller.step(data)

            # skip weight step: optizing \omega
            # but will updata subnet in finetuning phase
            if getattr(self.config.nas, "updata_weight", False) or self.controller.subnet != None:
                self.updata_weight_step(model, data)

            # measure elapsed time
            self.meters.batch_time.update(time.time() - end)

            # training logger
            if curr_step % 1 == 0:
                self.tb_logger.add_scalar('lr', current_lr, curr_step)

                if getattr(model, 'arch_parameters', False):
                    self.tb_logger.add_histogram('na_alphas', model.na_alphas, curr_step)
                    self.tb_logger.add_histogram('sc_alphas', model.sc_alphas, curr_step)
                    self.tb_logger.add_histogram('la_alphas', model.la_alphas, curr_step)

                remain_secs = (total_step - curr_step) * self.meters.batch_time.avg
                remain_time = datetime.timedelta(seconds=round(remain_secs))
                finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
                log_msg = f'Iter: [{curr_step}/{total_step}]\t' \
                        f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                        f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                        f'Remaining Time {remain_time} ({finish_time})'
                self.logger.info(log_msg)

                # update weight logs
                if getattr(self.config.nas, "updata_weight", False) or self.controller.subnet != None:
                    self.tb_logger.add_scalar('loss_train', self.meters.losses.avg, curr_step)
                    self.tb_logger.add_scalar('acc1_train', self.meters.top1.avg, curr_step)
                    self.tb_logger.add_scalar('acc5_train', self.meters.top5.avg, curr_step)

                    log_msg = f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                            f'Prec@1 {self.meters.top1.val:.3f} ({self.meters.top1.avg:.3f})\t' \
                            f'Prec@5 {self.meters.top5.val:.3f} ({self.meters.top5.avg:.3f})\t' \
                            f'LR {current_lr:.6f}\t'
                    self.logger.info(log_msg)

            end = time.time()

            # testing during training
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
                if getattr(model, 'arch_parameters', False):
                    self.state['arch_parameters'] = model.arch_parameters()

                torch.save(self.state, ckpt_name)
                genotype = model.genotype()
                self.logger.info('genotype = %s', genotype)

        self.dump_subnet_to_result()
        metrics = {}
        metrics['best_top1_val'] = best_prec1_val
        metrics['best_top1_test'] = best_prec1_test

        return metrics

    def dump_subnet_to_result(self):
        res = []
        res.append(f'genotype={self.model.genotype()}')
        result_filename =  os.path.join(self.path.result_path,
                        f'searched_result.txt')
        with open(result_filename, 'w+') as file:
            file.write('\n'.join(res))
            file.close()
        self.logger.info('searched res for {} saved in {}'.format(self.config.data.task, result_filename))

    # 测试一个超网，配置从self.config里面取
    def train(self):
        self._train(model=self.model)

    def evaluate(self):
        self._evaluate(model=self.model)
        prob_result_message = self.model.get_prob_result()
        for line in prob_result_message.split('\n'):
            self.logger.info(line)

    # 测试一个特定的子网，配置从self.subnet里面取
    def evaluate_subnet(self):
        self.subnet = self.controller.subnet
        assert self.subnet is not None

        self.save_subnet_weight = self.subnet.get('save_subnet_weight', False)

        if not getattr(self.subnet.subnet_settings, 'genotype', False):
            if getattr(self.subnet.subnet_settings, 'genotype_filename', False):
                result_line = open(self.subnet.subnet_settings.arch_filename, 'r').readlines()[-1]
                genotype = result_line.split('=')[1]
                self.subnet.subnet_settings.genotype = genotype
            else:
                # sample arch if self.subnet.subnet_settings.arch and arch_file is None
                subnet_settings = self.controller.sample_subnet_settings(sample_mode='random')
        else:
            subnet_settings = self.subnet.subnet_settings

        # build subnet from supernet
        subnet_model = self.controller.build_active_subnet(subnet_settings)

        # evaluate
        metrics = self._evaluate(model=subnet_model)

        # evaluate logging
        top1 = round(metrics['top1'], 3)
        subnet = {'subnet_settings': self.subnet.subnet_settings, 'top1': top1}
        self.logger.info('Subnet with settings: {}\ttop1 {}'.format(subnet_settings, top1))
        self.logger.info('Evaluate_subnet\t{}'.format(json.dumps(subnet)))

        # save weights
        if self.save_subnet_weight:
            state_dict = {}
            state_dict['model'] = subnet_model.state_dict()
            ckpt_name = f'{self.path.bignas_path}/ckpt_{top1}.pth.tar'
            torch.save(state_dict, ckpt_name)
        return subnet

    def build_finetune_dataset(self, max_epoch=None):
        if max_epoch is not None:
            self.config.data.max_epoch = max_epoch
        self.build_data()
        data = self.data['loader']
        data = save_load_split(data, gen_uniform_60_20_20_split)

    # finetune一个特定的子网，配置从self.subnet里面取
    def finetune_subnet(self):
        self.subnet = self.controller.subnet
        assert self.subnet is not None
        assert self.subnet.subnet_settings is not None

        if not getattr(self.subnet.subnet_settings, 'genotype', False):
            # read genotype from genotype_filename if self.subnet.subnet_settings.genotype_filename 
            # and genotype_filename is None
            if getattr(self.subnet.subnet_settings, 'genotype_filename', False):
                result_line = open(self.subnet.subnet_settings.genotype_filename, 'r').readlines()[-1]
                genotype = result_line.split('=')[1]
                self.subnet.subnet_settings.genotype = genotype
                subnet_settings = self.subnet.subnet_settings
            else:
                # sample arch if self.subnet.subnet_settings.genotype and genotype_filename is None
                subnet_settings = self.controller.sample_subnet_settings(sample_mode='random')
        else:
            subnet_settings = self.subnet.subnet_settings

        # build subnet from supernet
        subnet_model = self.controller.build_active_subnet(subnet_settings)

        # rebuild optimizer and scheduler
        self.optimizer = self._build_optimizer(self.subnet.optimizer, subnet_model)
        self.lr_scheduler = self._build_lr_scheduler(self.subnet.lr_scheduler, self.optimizer)
        self.build_finetune_dataset(max_epoch=self.subnet.data.max_epoch)

        # valiadate
        metrics = self._validate(model=subnet_model)
        top1_val = round(metrics['top1'], 3)

        # evaluate
        metrics = self._evaluate(model=subnet_model)
        top1_test = round(metrics['top1'], 3)

        # finetuneing logging
        subnet = {'subnet_settings': self.subnet.subnet_settings, 'top1_val': top1_val, 'top1_test': top1_test}
        self.logger.info('Before finetune subnet {}'.format(json.dumps(subnet)))

        # finetune restart
        last_iter = self.state['last_iter']
        last_epoch = self.state['last_epoch']
        self.state['last_iter'] = 0
        self.state['last_epoch'] = 0

        # finetuning 
        metrics = self._train(model=subnet_model)
        best_top1_val = metrics['best_top1_val']
        best_top1_test = metrics['best_top1_test']

        # record finetuning iterations
        self.state['last_iter'] = last_iter
        self.state['last_epoch'] = last_epoch

        # valiadate
        metrics = self._validate(model=subnet_model)
        top1_val = round(metrics['top1'], 3)

        # evaluate
        metrics = self._evaluate(model=subnet_model)
        top1_test = round(metrics['top1'], 3)

        # finetuneing logging
        subnet = {'subnet_settings': self.subnet.subnet_settings, 'top1_val': top1_val, 'top1_test': top1_test, 'best_top1_val': best_top1_val, 'best_top1_test': best_top1_test}
        self.logger.info('After finetune subnet {}'.format(json.dumps(subnet)))
        return subnet

def main():
    parser = argparse.ArgumentParser(description='Graph Neural archtecture coding Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--phase', default='train_search')

    args = parser.parse_args()
    # build solver
    solver = NACInductiveSolver(args.config)

    # evaluate or fintune or train_search
    if args.phase in ['evaluate_subnet', 'finetune_subnet']:
        if not hasattr(solver.config.saver, 'pretrain'):
            solver.logger.warn(f'{args.phase} without resuming any solver checkpoints.')
        if args.phase == 'evaluate_subnet':
            solver.evaluate_subnet()
        else:
            solver.finetune_subnet()
    elif args.phase == 'train_search':
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