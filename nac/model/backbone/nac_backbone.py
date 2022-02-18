import torch.nn as nn
import torch.nn.functional as F

from ..ops import NaOp, ScOp, LaOp

class NACBackBone(nn.Module):
    '''
        implement this for sane.
        Actually, sane can be seen as the combination of three cells, node aggregator, skip connection, and layer aggregator
        for sane, we dont need cell, since the DAG is the whole search space, and what we need to do is implement the DAG.
    '''
    def __init__(self, genotype, in_dim, out_dim, hidden_size, num_layers=3, in_dropout=0.5, out_dropout=0.5, act='relu', config=None, jk=False):
        super(NACBackBone, self).__init__()
        self.arch = genotype
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.jk = jk
        self.num_layers = num_layers
        self.in_dropout = in_dropout
        self.out_dropout = out_dropout
        ops = genotype.split('||')
        self.config = config

        # node aggregator op
        self.lin1 = nn.Linear(in_dim, hidden_size)
        self.gnn_layers = nn.ModuleList(
                [NaOp(ops[i], hidden_size, hidden_size, act, with_linear=config.with_linear) for i in range(num_layers)])

        if self.jk:
            # skip op
            if self.config.fix_last:
                if self.num_layers > 1:
                    self.sc_layers = nn.ModuleList([ScOp(ops[i+num_layers]) for i in range(num_layers - 1)])
                else:
                    self.sc_layers = nn.ModuleList([ScOp(ops[num_layers])])
            else:
                # no output conditions.
                skip_op = ops[num_layers:2 * num_layers]
                if skip_op == ['none'] * num_layers:
                    skip_op[-1] = 'skip'
                    print('skip_op:', skip_op)
                self.sc_layers = nn.ModuleList([ScOp(skip_op[i]) for i in range(num_layers)])

            #layer aggregator op
            self.layer6 = LaOp(ops[-1], hidden_size, 'linear', num_layers)
        self.classifier = nn.Linear(hidden_size, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.lin1(x)
        x = F.dropout(x, p=self.in_dropout, training=self.training)
        js = []

        for i in range(self.num_layers):
            x = self.gnn_layers[i](x, edge_index)
            if self.config.with_layernorm:
                layer_norm = nn.LayerNorm(normalized_shape=x.size(), elementwise_affine=False)
                x = layer_norm(x)
            x = F.dropout(x, p=self.in_dropout, training=self.training)
            if self.jk:
                if i == self.num_layers - 1 and self.config.fix_last:
                    js.append(x)
                else:
                    js.append(self.sc_layers[i](x))

        if self.jk:
            x5 = self.layer6(js)
            x5 = F.dropout(x5, p=self.out_dropout, training=self.training)
            logits = self.classifier(x5)
        else:
            logits = self.classifier(x)

        return logits

    def genotype(self):
        return self.arch