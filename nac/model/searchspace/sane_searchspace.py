import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..ops.dynamic_layers import NaMixedOp, ScMixedOp, LaMixedOp
from ..ops.dynamic_layers import NA_PRIMITIVES, SC_PRIMITIVES, LA_PRIMITIVES
from ..backbone.nac_backbone import NACBackBone

class SANESearchSpace(nn.Module):
    '''
        implement this for sane.
        Actually, sane can be seen as the combination of three cells, node aggregator, skip connection, and layer aggregator
        for sane, we dont need cell, since the DAG is the whole search space, and what we need to do is implement the DAG.
    '''

    def __init__(self, in_dim, out_dim, hidden_size,
                num_layers=3,
                dropout=0.5,
                epsilon=0.0,
                with_conv_linear=False,
                fix_last=False,
                _NA_PRIMITIVES=NA_PRIMITIVES,
                _SC_PRIMITIVES=SC_PRIMITIVES,
                _LA_PRIMITIVES=LA_PRIMITIVES,
                with_bn=False
                ):
        super(SANESearchSpace, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout=dropout
        self.epsilon = epsilon
        self.explore_num = 0
        self.with_linear = with_conv_linear
        self.fix_last = fix_last
        self.with_bn = with_bn

        self.NA_PRIMITIVES = _NA_PRIMITIVES
        self.SC_PRIMITIVES = _SC_PRIMITIVES
        self.LA_PRIMITIVES = _LA_PRIMITIVES

        #node aggregator op
        self.lin1 = nn.Linear(in_dim, hidden_size)
        self.layer1 = NaMixedOp(hidden_size, hidden_size, self.with_linear, self.NA_PRIMITIVES, with_bn=with_bn)
        self.layer2 = NaMixedOp(hidden_size, hidden_size, self.with_linear, self.NA_PRIMITIVES, with_bn=with_bn)
        self.layer3 = NaMixedOp(hidden_size, hidden_size, self.with_linear, self.NA_PRIMITIVES, with_bn=with_bn)

        #skip op
        self.layer4 = ScMixedOp(self.SC_PRIMITIVES)
        self.layer5 = ScMixedOp(self.SC_PRIMITIVES)
        if self.fix_last:
            self.layer6 = ScMixedOp(self.SC_PRIMITIVES)

        #layer aggregator op
        self.layer7 = LaMixedOp(hidden_size, num_layers, self.LA_PRIMITIVES, with_bn=with_bn)
        self.classifier = nn.Linear(hidden_size, out_dim)
        self._initialize_alphas()

    def forward(self, data, discrete=False):
        x, edge_index = data.x, data.edge_index

        self.na_weights = F.softmax(self.na_alphas, dim=-1)
        self.sc_weights = F.softmax(self.sc_alphas, dim=-1)
        self.la_weights = F.softmax(self.la_alphas, dim=-1)

        # generate weights by softmax
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x1 = self.layer1(x, self.na_weights[0], edge_index)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.layer2(x1, self.na_weights[1], edge_index)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x3 = self.layer3(x2, self.na_weights[2], edge_index)
        x3 = F.dropout(x3, p=self.dropout, training=self.training)

        if self.fix_last:
            x4 = (self.layer4(x1, self.sc_weights[0]), self.layer5(x2, self.sc_weights[1]), self.layer6(x3, self.sc_weights[2]))
        else:
            x4 = (x3, self.layer4(x1, self.sc_weights[0]), self.layer5(x2, self.sc_weights[1]))

        x5 = self.layer7(x4, self.la_weights[0])
        x5 = F.dropout(x5, p=self.dropout, training=self.training)

        logits = self.classifier(x5)
        return logits

    def _initialize_alphas(self):
        num_na_ops = len(self.NA_PRIMITIVES)
        num_sc_ops = len(self.SC_PRIMITIVES)
        num_la_ops = len(self.LA_PRIMITIVES)

        self.na_alphas = Variable(1e-3*torch.randn(self.num_layers, num_na_ops), requires_grad=True)
        if self.fix_last:
            self.sc_alphas = Variable(1e-3*torch.randn(self.num_layers - 1, num_sc_ops), requires_grad=True)
        else:
            self.sc_alphas = Variable(1e-3*torch.randn(self.num_layers, num_sc_ops), requires_grad=True)
        self.la_alphas = Variable(1e-3*torch.randn(1, num_la_ops), requires_grad=True)
        self._arch_parameters = [
            self.na_alphas,
            self.sc_alphas,
            self.la_alphas,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def get_alpha_result(self):
        def _parse(na_weights):
            gene = []
            _, na_indices = torch.topk(na_weights, k=5, dim=-1, largest=True)
            for i, na_indice in enumerate(na_indices):
                gene.append([])
                for k in na_indice:
                    gene[i].append(self.NA_PRIMITIVES[int(k)])
                gene[i] = str(gene[i])
            return gene
        gene = _parse(F.softmax(self.na_alphas, dim=-1).detach())
        return '[' + ','.join(gene) + ']'

    def get_prob_result(self):
        result = '*' * 20 + 'Alpha parameter' + '*' * 20 + '\n'
        result = self.genotype() + '\n'
        result += str(self.NA_PRIMITIVES) + '\n'
        result += str(self.na_alphas) + '\n'
        result += str(self.SC_PRIMITIVES) + '\n'
        result += str(self.sc_alphas) + '\n'
        result += str(self.LA_PRIMITIVES) + '\n'
        result += str(self.la_alphas) + '\n'
        result += self.get_alpha_result()
        return result

    def genotype(self):
        def _parse(na_weights, sc_weights, la_weights):
            gene = []
            na_indices = torch.argmax(na_weights, dim=-1)
            for k in na_indices:
                gene.append(self.NA_PRIMITIVES[k])
            sc_indices = torch.argmax(sc_weights, dim=-1)
            for k in sc_indices:
                gene.append(self.SC_PRIMITIVES[k])
            #la_indices = la_weights.argmax(dim=-1)
            la_indices = torch.argmax(la_weights, dim=-1)
            for k in la_indices:
                gene.append(self.LA_PRIMITIVES[k])
            return '||'.join(gene)

        gene = _parse(F.softmax(self.na_alphas, dim=-1).data.cpu(), F.softmax(self.sc_alphas, dim=-1).data.cpu(), F.softmax(self.la_alphas, dim=-1).data.cpu())
        return gene

    def sample_genotype(self):
        gene = []
        for _ in range(3):
            op = np.random.choice(self.NA_PRIMITIVES, 1)[0]
            gene.append(op)
        for _ in range(2):
            op = np.random.choice(self.SC_PRIMITIVES, 1)[0]
            gene.append(op)
        op = np.random.choice(self.LA_PRIMITIVES, 1)[0]
        gene.append(op)
        return '||'.join(gene)

    def get_weights_from_arch(self, arch):
        arch_ops = arch.split('||')
        #print('arch=%s' % arch)
        num_na_ops = len(self.NA_PRIMITIVES)
        num_sc_ops = len(self.SC_PRIMITIVES)
        num_la_ops = len(self.LA_PRIMITIVES)

        na_alphas = Variable(torch.zeros(3, num_na_ops), requires_grad=True)
        sc_alphas = Variable(torch.zeros(2, num_sc_ops), requires_grad=True)
        la_alphas = Variable(torch.zeros(1, num_la_ops), requires_grad=True)

        for i in range(3):
            ind = self.NA_PRIMITIVES.index(arch_ops[i])
            na_alphas[i][ind] = 1

        for i in range(3, 5):
            ind = self.SC_PRIMITIVES.index(arch_ops[i])
            sc_alphas[i-3][ind] = 1

        ind = self.LA_PRIMITIVES.index(arch_ops[5])
        la_alphas[0][ind] = 1

        arch_parameters = [na_alphas, sc_alphas, la_alphas]
        return arch_parameters

    def set_model_weights(self, weights):
        self.na_weights = weights[0]
        self.sc_weights = weights[1]
        self.la_weights = weights[2]

    def sample_active_subnet(self, sample_mode='random', subnet_settings=None):
        # In the forward pass, we need to clarify the sample mode ['random', 'subnet']
        if sample_mode == 'random':
            genotype = self.sample_genotype()
            subnet_settings.genotype = genotype
            return subnet_settings

    def build_active_subnet(self, subnet_settings):
        subnet = NACBackBone(subnet_settings['genotype'],
                                self.in_dim, self.out_dim, subnet_settings['hidden_size'], 
                                self.num_layers, in_dropout=subnet_settings['in_dropout'],
                                out_dropout=subnet_settings['out_dropout'],
                                act=subnet_settings['act'],
                                config=subnet_settings['config'])
        return subnet

if __name__ == '__main__':
    pass