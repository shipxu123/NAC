import torch
import torch.nn as nn
from .normal_ops import *

NA_PRIMITIVES = ['mlp',
    'sage',
    'sage_sum',
    'sage_max',
    'gcn',
    'gat',
    'gat_sym',
    'gat_linear',
    'gat_cos',
    'gat_generalized_linear',
    'geniepath',
    'chebconv',
    'chebconv_1',
    'chebconv_2',
    'chebconv_3',
    'chebconv_4',
    'chebconv_5',
    'chebconv_6',
    'chebconv_7',
    'gin',
    'gin_2',
    'gin_1',
    'gin_0',
    'gin_-1',
    'gin_-2',
    'gin_trainable'
]

SC_PRIMITIVES=[
    'none',
    'skip',
]
LA_PRIMITIVES=[
    'l_max',
    'l_concat',
    'l_lstm'
]

class NaMixedOp(nn.Module):

    def __init__(self, in_dim, out_dim, with_linear, _NA_PRIMITIVES=NA_PRIMITIVES, with_bn=False):
        super(NaMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.with_linear = with_linear

        for primitive in _NA_PRIMITIVES:
            op = NA_OPS[primitive](in_dim, out_dim, with_bn=with_bn)
            self._ops.append(op)

            if with_linear:
                self._ops_linear = nn.ModuleList()
                op_linear = torch.nn.Linear(in_dim, out_dim)
                self._ops_linear.append(op_linear)

    def forward(self, x, weights, edge_index):
        mixed_res = []
        if self.with_linear:
            for w, op, linear in zip(weights, self._ops, self._ops_linear):
                mixed_res.append(w * F.elu(op(x, edge_index)+linear(x)))
        else:
            for w, op in zip(weights, self._ops):
                mixed_res.append(w * F.elu(op(x, edge_index)))
        return sum(mixed_res)


class ScMixedOp(nn.Module):

    def __init__(self, _SC_PRIMITIVES=SC_PRIMITIVES):
        super(ScMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in _SC_PRIMITIVES:
            op = SC_OPS[primitive]()
            self._ops.append(op)

    def forward(self, x, weights):
        mixed_res = []
        for w, op in zip(weights, self._ops):
            mixed_res.append(w * op(x))
        return sum(mixed_res)

class LaMixedOp(nn.Module):

    def __init__(self, hidden_size, num_layers=None, _LA_PRIMITIVES=LA_PRIMITIVES, with_bn=False):
        super(LaMixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in _LA_PRIMITIVES:
            op = LA_OPS[primitive](hidden_size, num_layers, with_bn=with_bn)
            self._ops.append(op)

    def forward(self, x, weights):
        mixed_res = []
        for w, op in zip(weights, self._ops):
            mixed_res.append(w * F.relu(op(x)))
        return sum(mixed_res)