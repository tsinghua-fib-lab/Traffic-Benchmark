import torch
import torch.nn as nn
from model.AGCRNCell import AGCRNCell

class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

import sys
class AGCRN(nn.Module):
    def __init__(self, args):
        super(AGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers)

        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        # source = source[:, :1, :, :]
        # source = source.transpose(1, 3)
        print(source.shape)
        print('0: ', torch.sum(torch.isnan(source)))
        init_state = self.encoder.init_hidden(source.shape[0])
        print('1: ', torch.sum(torch.isnan(init_state)))

        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        print('2: ', torch.sum(torch.isnan(output)))
        print('2 node embedding: ', torch.sum(torch.isnan(self.node_embeddings)))

        output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        print('3: ', torch.sum(torch.isnan(output)))

        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        print('4: ', torch.sum(torch.isnan(output)))

        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C
        print('5: ', torch.sum(torch.isnan(output)))
        sys.exit(0)

        # torch.Size([64, 12, 207, 1])
        # 0: tensor(0, device='cuda:0')
        # 1: tensor(0)
        # 2: tensor(10174464, device='cuda:0')
        # 2 node embedding: tensor(0, device='cuda:0')
        # 3: tensor(847872, device='cuda:0')
        # 4: tensor(158976, device='cuda:0')
        # 5: tensor(158976, device='cuda:0')

        # torch.Size([64, 12, 207, 2])
        # 0: tensor(0, device='cuda:1')
        # 1: tensor(0)
        # 2: tensor(0, device='cuda:1')
        # 2
        # node
        # embedding: tensor(0, device='cuda:1')
        # 3: tensor(0, device='cuda:1')
        # 4: tensor(0, device='cuda:1')
        # 5: tensor(0, device='cuda:1')
        return output