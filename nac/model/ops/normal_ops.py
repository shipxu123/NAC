import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, JumpingKnowledge, GINConv, ChebConv

from .layer import GeoLayer, GeniePathLayer, MLPLayer

NA_OPS = {
    'mlp': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'mlp', with_bn),
    # gcn family
    'sage': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'sage', with_bn),
    'sage_sum': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'sum', with_bn),
    'sage_max': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'max', with_bn),
    'gcn': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'gcn', with_bn),
    'gat': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'gat', with_bn),
    'gat_sym': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'gat_sym', with_bn),
    'gat_linear': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'linear', with_bn),
    'gat_cos': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'cos', with_bn),
    'gat_generalized_linear': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'generalized_linear', with_bn),
    'geniepath': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'geniepath', with_bn),
    # high frequence
    # cheb conv
    'chebconv': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'chebconv_5', with_bn),
    'chebconv_1': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'chebconv_1', with_bn),
    'chebconv_2': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'chebconv_2', with_bn),
    'chebconv_3': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'chebconv_3', with_bn),
    'chebconv_4': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'chebconv_4', with_bn),
    'chebconv_5': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'chebconv_5', with_bn),
    'chebconv_6': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'chebconv_6', with_bn),
    'chebconv_7': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'chebconv_7', with_bn),
    # gin conv
    'gin': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'gin_0', with_bn),
    'gin_2': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'gin_2', with_bn),
    'gin_1': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'gin_1', with_bn),
    'gin_0': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'gin_0', with_bn),
    'gin_-1': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'gin_-1', with_bn),
    'gin_-2': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'gin_-2', with_bn),
    'gin_trainable': lambda in_dim, out_dim, with_bn=False: NaAggregator(in_dim, out_dim, 'gin_trainable', with_bn),
}

SC_OPS={
    'none': lambda: Zero(),
    'skip': lambda: Identity(),
}

LA_OPS={
    'l_max': lambda hidden_size, num_layers, with_bn=False: LaAggregator('max', hidden_size, num_layers, with_bn),
    'l_concat': lambda hidden_size, num_layers, with_bn=False: LaAggregator('cat', hidden_size, num_layers, with_bn),
    'l_lstm': lambda hidden_size, num_layers, with_bn=False: LaAggregator('lstm', hidden_size, num_layers, with_bn)
}


class NaAggregator(nn.Module):

    def __init__(self, in_dim, out_dim, aggregator, with_bn=False):
        super(NaAggregator, self).__init__()
        self.with_bn = with_bn
        if with_bn:
            self.bn = nn.BatchNorm1d(int(out_dim))

        if 'mlp' == aggregator:
            self._op = MLPLayer(in_dim, out_dim)
        elif 'sage' == aggregator:
            self._op = SAGEConv(in_dim, out_dim, normalize=True)
        elif aggregator in ['sum', 'max']:
            self._op = GeoLayer(in_dim, out_dim, att_type='const', agg_type=aggregator, dropout=0.5)
        elif aggregator in ['geniepath']:
            self._op = GeniePathLayer(in_dim, out_dim)
        elif 'gcn' == aggregator:
            self._op = GCNConv(in_dim, out_dim)
        elif 'gat' == aggregator:
            heads = 8
            out_dim /= heads
            self._op = GATConv(in_dim, int(out_dim), heads=heads, dropout=0.5)
        elif aggregator in ['gat_sym', 'cos', 'linear', 'generalized_linear']:
            heads = 8
            out_dim /= heads
            self._op = GeoLayer(in_dim, int(out_dim), heads=heads, att_type=aggregator, dropout=0.5)
        elif 'chebconv' in aggregator:
            k = int(aggregator.split('_')[-1])
            self._op = ChebConv(in_dim, out_dim, K=k)
        elif 'gin' in aggregator:
            nn1 = Sequential(Linear(in_dim, out_dim), ReLU(), Linear(out_dim, out_dim))
            if 'gin_trainable' == aggregator:
                self._op = GINConv(nn1, train_eps=False)
            else:
                eps = float(aggregator.split('_')[-1])
                self._op = GINConv(nn1, eps=eps)
        else:
            raise NotImplementedError(f'{aggregator} is not supported!')

    def forward(self, x, edge_index):
        if self.with_bn:
            x = self._op(x, edge_index)
            return self.bn(x)
        else:
            return self._op(x, edge_index)


class LaAggregator(nn.Module):

    def  __init__(self, mode, hidden_size, num_layers=3, with_bn=False):
        super(LaAggregator, self).__init__()
        self.with_bn = with_bn
        if with_bn:
            self.bn = nn.BatchNorm1d(int(hidden_size))

        self.jump = JumpingKnowledge(mode, channels=hidden_size, num_layers=num_layers)
        if mode == 'cat':
            self.lin = Linear(hidden_size * num_layers, hidden_size)
        else:
            self.lin = Linear(hidden_size, hidden_size)

    def forward(self, xs):
        if self.with_bn:
            x = self.lin(F.relu(self.jump(xs)))
            return self.bn(x)
        else:
            return self.lin(F.relu(self.jump(xs)))


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x.mul(0.)