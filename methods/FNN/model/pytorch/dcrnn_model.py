import numpy as np
import torch
import torch.nn as nn

from model.pytorch.dcrnn_cell import DCGRUCell

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(model_kwargs.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get('filter_type', 'laplacian')
        self.num_nodes = int(model_kwargs.get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.output_dim = int(model_kwargs.get('output_dim', 1))
        self.horizon = int(model_kwargs.get('horizon', 1))  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None):
        """
        Decoder forward pass.

        :param inputs: shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class DCRNNModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, logger, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.encoder_model = EncoderModel(adj_mx, **model_kwargs)
        self.decoder_model = DecoderModel(adj_mx, **model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))
        self._logger = logger

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        :param labels: (self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_seen: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, labels=None, batches_seen=None):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_seen: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        encoder_hidden_state = self.encoder(inputs)
        self._logger.debug("Encoder complete, starting decoder")
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        self._logger.debug("Decoder complete")
        if batches_seen == 0:
            self._logger.info(
                "Total trainable parameters {}".format(count_parameters(self))
            )
        return outputs


####################################################################################################################


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,
                               num_timesteps_output)
        self.num_nodes = num_nodes
        self.input_dim = num_features
        self.seq_len = num_timesteps_input
        self.horizon = num_timesteps_output

    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        X = X.view(self.seq_len, -1, self.num_nodes, self.input_dim).permute(1, 2, 0, 3).contiguous()
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4.permute(2, 0, 1).contiguous()

        # :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
        #          y: shape (horizon, batch_size, num_sensor * output_dim)

####################################################################################################################

import math
import random
from typing import List, Tuple

import numpy as np
import dgl
import torch
from dgl import DGLGraph, init
from torch import nn, Tensor


class MultiLayerPerception(nn.Sequential):
    def __init__(self, hiddens: List[int], hidden_act, out_act: bool):
        super(MultiLayerPerception, self).__init__()
        for i in range(1, len(hiddens)):
            self.add_module(f'Layer{i}', nn.Linear(hiddens[i - 1], hiddens[i]))
            if i < len(hiddens) - 1 or out_act:
                self.add_module(f'Activation{i}', hidden_act())


class MetaDense(nn.Module):
    def __init__(self, f_in: int, f_out: int, feat_size: int, meta_hiddens: List[int]):
        super(MetaDense, self).__init__()
        self.weights_mlp = MultiLayerPerception([feat_size] + meta_hiddens + [f_in * f_out], nn.Sigmoid, False)
        self.bias_mlp = MultiLayerPerception([feat_size] + meta_hiddens + [f_out], nn.Sigmoid, False)

    def forward(self, feature: Tensor, data: Tensor) -> Tensor:
        """
        :param feature: tensor, [N, F]
        :param data: tensor, [B, N, F_in]
        :return: tensor, [B, N, F_out]
        """
        b, n, f_in = data.shape
        data = data.reshape(b, n, 1, f_in)
        weights = self.weights_mlp(feature).reshape(1, n, f_in, -1)  # [F_in, F_out]
        bias = self.bias_mlp(feature)  # [n, F_out]

        return data.matmul(weights).squeeze(2) + bias


class RNNCell(nn.Module):
    def __init__(self):
        super(RNNCell, self).__init__()

    def one_step(self, feature: Tensor, data: Tensor, begin_state: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        :param feature: tensor, [N, F]
        :param data: tensor, [B, N, F]
        :param begin_state: None or tensor, [B, N, F]
        :return: output, tensor, [B, N, F]
                begin_state, [B, N, F]
        """
        raise NotImplementedError("Not Implemented")

    def forward(self, feature: Tensor, data: Tensor, begin_state: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        :param feature: tensor, [N, F]
        :param data: tensor, [B, T, N, F]
        :param begin_state: [B, N, F]
        :return:
        """
        b, t, n, _ = data.shape

        outputs, state = list(), begin_state
        for i_t in range(t):
            output, state = self.one_step(feature, data[:, i_t], state)
            outputs.append(output)

        return torch.stack(outputs, 1), state


class MetaGRUCell(RNNCell):
    def __init__(self, f_in: int, hid_size: int, feat_size: int, meta_hiddens: List[int]):
        super(MetaGRUCell, self).__init__()
        self.hidden_size = hid_size
        self.dense_zr = MetaDense(f_in + hid_size, 2 * hid_size, feat_size, meta_hiddens=meta_hiddens)

        self.dense_i2h = MetaDense(f_in, hid_size, feat_size, meta_hiddens=meta_hiddens)
        self.dense_h2h = MetaDense(hid_size, hid_size, feat_size, meta_hiddens=meta_hiddens)

    def one_step(self, feature: Tensor, data: Tensor, begin_state: Tensor = None) -> Tuple[Tensor, Tensor]:
        b, n, _ = data.shape
        if begin_state is None:
            begin_state = torch.zeros(b, n, self.hidden_size, dtype=data.dtype, device=data.device)

        data_and_state = torch.cat([data, begin_state], -1)
        zr = torch.sigmoid(self.dense_zr(feature, data_and_state))
        z, r = zr.split(self.hidden_size, -1)

        state = z * begin_state + (1 - z) * torch.tanh(self.dense_i2h(feature, data) + self.dense_h2h(feature, r * begin_state))


        # c = torch.tanh(self.dense_i2h(feature, data))
        # h = self.dense_h2h(feature, r * begin_state)
        #
        # state = z * begin_state + torch.sub(1., z) * c + h
        return state, state


class NormalGRUCell(RNNCell):
    def __init__(self, f_in: int, hid_size: int):
        super(NormalGRUCell, self).__init__()
        self.cell = nn.GRUCell(f_in, hid_size)

    def one_step(self, feature: Tensor, data: Tensor, begin_state: Tensor = None) -> Tuple[Tensor, Tensor]:
        b, n, _ = data.shape
        data = data.reshape(b * n, -1)
        if begin_state is not None:
            begin_state = begin_state.reshape(b * n, -1)
        h = self.cell(data, begin_state)
        h = h.reshape(b, n, -1)
        return h, h

import sys
class GraphAttNet(nn.Module):
    def __init__(self, dist: np.ndarray, edge: list, hid_size: int, feat_size: int, meta_hiddens: List[int]):
        super(GraphAttNet, self).__init__()
        self.hidden_size = hid_size
        self.feature_size = feat_size
        self.meta_hiddens = meta_hiddens

        self.num_nodes = n = dist.shape[0]
        src, dst, dis = list(), list(), list()
        for i in range(n):
            for j in edge[i]:
                src.append(j)
                dst.append(i)
                dis.append(dist[j, i])

        dist = torch.tensor(dis).unsqueeze_(1)
        g = DGLGraph()
        g.set_n_initializer(init.zero_initializer)
        g.add_nodes(n)
        g.add_edges(src, dst, {'dist': dist})
        self.graph = g

    def forward(self, state: Tensor, feature: Tensor) -> Tensor:
        """
        :param state: tensor, [B, T, N, F] or [B, N, F]
        :param feature: tensor, [N, F]
        :return: tensor, [B, T, N, F]
        """
        # print(state.shape)
        # torch.Size([32, 12, 207, 32])

        # shape => [N, B, T, F] or [N, B, F]
        state = state.unsqueeze(0).transpose(0, -2).squeeze(-2)
        g = self.graph.local_var()
        g.to(state.device)
        g.ndata['state'] = state
        g.ndata['feature'] = feature
        g.update_all(self.msg_edge, self.msg_reduce)
        state = g.ndata.pop('new_state')
        # print(state.shape)
        # torch.Size([207, 32, 12, 32])
        # sys.exit(0)
        return state.unsqueeze(-2).transpose(0, -2).squeeze(0)

    def msg_edge(self, edge: dgl.EdgeBatch):
        """
        :param edge: a dictionary of edge data.
            edge.src['state'] and edge.dst['state']: hidden states of the nodes, with shape [e, b, t, d] or [e, b, d]
            edge.src['feature'] and edge.dst['state']: features of the nodes, with shape [e, d]
            edge.data['dist']: distance matrix of the edges, with shape [e, d]
        :return: a dictionray of messages
        """
        raise NotImplementedError('Not implemented.')

    def msg_reduce(self, node: dgl.NodeBatch):
        """
        :param node:
                node.mailbox['state'], tensor, [n, e, b, t, d] or [n, e, b, d]
                node.mailbox['alpha'], tensor, [n, e, b, t, d] or [n, e, b, d]
        :return: tensor, [n, b, t, d] or [n, b, d]
        """
        raise NotImplementedError('Not implemented.')


class MetaGAT(GraphAttNet):
    def __init__(self, *args, **kwargs):
        super(MetaGAT, self).__init__(*args, **kwargs)
        self.w_mlp = MultiLayerPerception(
            [self.feature_size * 2 + 1] + self.meta_hiddens + [self.hidden_size * 2 * self.hidden_size],
            nn.Sigmoid, False)
        self.act = nn.LeakyReLU()
        self.weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def msg_edge(self, edge: dgl.EdgeBatch):
        state = torch.cat([edge.src['state'], edge.dst['state']], -1)  # [X, B, T, 2H] or [X, B, 2H]
        feature = torch.cat([edge.src['feature'], edge.dst['feature'], edge.data['dist']], -1)  # [X, 2F + 1]

        weight = self.w_mlp(feature).reshape(-1, self.hidden_size * 2, self.hidden_size)  # [X, 2H, H]

        shape = state.shape
        state = state.reshape(shape[0], -1, shape[-1])

        # [X, ?, 2H] * [X. 2H, H] => [X, ?, H]
        alpha = self.act(torch.bmm(state, weight))

        alpha = alpha.reshape(*shape[:-1], self.hidden_size)
        return {'alpha': alpha, 'state': edge.src['state']}

    def msg_reduce(self, node: dgl.NodeBatch):
        state = node.mailbox['state']
        alpha = node.mailbox['alpha']
        alpha = torch.softmax(alpha, 1)

        new_state = torch.relu(torch.sum(alpha * state, dim=1)) * torch.sigmoid(self.weight)
        return {'new_state': new_state}


class STMetaEncoder(nn.Module):
    def __init__(self, input_dim: int, rnn_types: List[str], rnn_hiddens: List[int], feat_size: int,
                 meta_hiddens: List[int], graph: Tuple[np.ndarray, list, list]):
        super(STMetaEncoder, self).__init__()

        dist, e_in, e_out = graph

        grus, gats = list(), list()

        rnn_hiddens = [input_dim] + rnn_hiddens
        for i, rnn_type in enumerate(rnn_types):
            in_dim, out_dim = rnn_hiddens[i], rnn_hiddens[i + 1]
            if rnn_type == 'NormalGRU':
                grus.append(NormalGRUCell(in_dim, out_dim))
            elif rnn_type == 'MetaGRU':
                grus.append(MetaGRUCell(in_dim, out_dim, feat_size, meta_hiddens))
            else:
                raise ValueError(f'{rnn_type} is not implemented.')

            if i == len(rnn_types) - 1:
                break

            g1 = MetaGAT(dist.T, e_in, out_dim, feat_size, meta_hiddens)
            g2 = MetaGAT(dist, e_out, out_dim, feat_size, meta_hiddens)
            gats.append(nn.ModuleList([g1, g2]))

        self.grus = nn.ModuleList(grus)
        self.gats = nn.ModuleList(gats)

    def forward(self, feature: Tensor, data: Tensor) -> List[Tensor]:
        """
        :param feature: tensor, [N, F]
        :param data: tensor, [B, T, N, F]
        :return: list of tensors
        """
        states = list()
        for depth, (g1, g2) in enumerate(self.gats):
            data, state = self.grus[depth](feature, data)
            states.append(state)
            data = g1(data, feature) + g2(data, feature)
        else:
            _, state = self.grus[-1](feature, data)
            states.append(state)
        return states


class STMetaDecoder(nn.Module):
    def __init__(self, n_preds: int, output_dim: int, rnn_types: List[str], rnn_hiddens: List[int], feat_size: int,
                 meta_hiddens: List[int], graph: Tuple[np.ndarray, list, list], input_dim):
        super(STMetaDecoder, self).__init__()
        self.output_dim = output_dim
        self.n_preds = n_preds

        dist, e_in, e_out = graph

        grus, gats = list(), list()

        # rnn_hiddens = [output_dim] + rnn_hiddens
        rnn_hiddens = [input_dim] + rnn_hiddens
        self.input_dim = input_dim
        for i, rnn_type in enumerate(rnn_types):
            in_dim, out_dim = rnn_hiddens[i], rnn_hiddens[i + 1]
            if rnn_type == 'NormalGRU':
                grus.append(NormalGRUCell(in_dim, out_dim))
            elif rnn_type == 'MetaGRU':
                grus.append(MetaGRUCell(in_dim, out_dim, feat_size, meta_hiddens))
            else:
                raise ValueError(f'{rnn_type} is not implemented.')

            if i == len(rnn_types) - 1:
                break

            g1 = MetaGAT(dist.T, e_in, out_dim, feat_size, meta_hiddens)
            g2 = MetaGAT(dist, e_out, out_dim, feat_size, meta_hiddens)
            gats.append(nn.ModuleList([g1, g2]))

        self.grus = nn.ModuleList(grus)
        self.gats = nn.ModuleList(gats)
        self.out = nn.Linear(rnn_hiddens[1], output_dim)

    # def sampling(self):
    #     """ Schedule sampling: sampling the ground truth. """
    #     threshold = self.cl_decay_steps / (self.cl_decay_steps + math.exp(self.global_steps / self.cl_decay_steps))
    #     return float(random.random() < threshold)

    def forward(self, feature: Tensor, begin_states: List[Tensor], targets: Tensor = None,
                teacher_force: bool = 0.5) -> Tensor:
        """
        :param feature: tensor, [N, F]
        :param begin_states: list of tensors, each of [B, N, hidden_size]
        :param targets: none or tensor, [B, T, N, input_dim]
        :param teacher_force: float, random to use targets as decoder inputs
        :return:
        """
        b, n, _ = begin_states[0].shape
        aux = targets[:,:,:, self.output_dim:] # [b,t,n,d]
        label = targets[:,:,:, :self.output_dim] # [b,t,n,d]
        go = torch.zeros(b, n, self.input_dim, device=feature.device, dtype=feature.dtype)

        # outputs = list()
        outputs, states = [], begin_states
        for i_pred in range(self.n_preds):
            if i_pred == 0:
                inputs = go

            for depth, (g1, g2) in enumerate(self.gats):
                inputs, states[0] = self.grus[depth].one_step(feature, inputs, states[0])
                inputs = (g1(inputs, feature) + g2(inputs, feature)) / 2
            else:
                # print(len(self.grus), len(states))
                inputs, states[1] = self.grus[-1].one_step(feature, inputs, states[1])
            inputs = self.out(inputs)
            outputs.append(inputs)
            if self.training and (targets is not None) and (random.random() < teacher_force):
                # inputs = targets[:, i_pred]
                inputs = label[:, i_pred]
            inputs = torch.cat([inputs, aux[:, i_pred, :, :]], -1)
        return torch.stack(outputs, 1)


class STMetaNet(nn.Module):
    def __init__(self,
                 graph: Tuple[np.ndarray, list, list],
                 n_preds: int,
                 input_dim: int,
                 output_dim: int,
                 cl_decay_steps: int,
                 rnn_types: List[str],
                 rnn_hiddens: List[int],
                 meta_hiddens: List[int],
                 geo_hiddens: List[int]):
        super(STMetaNet, self).__init__()
        feat_size = geo_hiddens[-1]
        self.cl_decay_steps = cl_decay_steps
        self.encoder = STMetaEncoder(input_dim, rnn_types, rnn_hiddens, feat_size, meta_hiddens, graph)
        self.decoder = STMetaDecoder(n_preds, output_dim, rnn_types, rnn_hiddens, feat_size, meta_hiddens, graph, input_dim)
        self.geo_encoder = MultiLayerPerception(geo_hiddens, hidden_act=nn.ReLU, out_act=True)

        features = graph[0]
        self.num_nodes = features.shape[0]
        # self.num_nodes = 500
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = 12
        self.horizon = n_preds

        self.fc1 = nn.Linear(input_dim * self.seq_len, 256)
        self.fc2 = nn.Linear(256, self.horizon * self.output_dim)



# FNN Feed forward neural network with two hidden layers, each layer contains 256 units. The initial learning rate is 1e−3, and reduces to 1 10every 20 epochs starting at the 50th epochs.
# In addition, for all hidden layers, dropout with ratio 0.5 and L2 weight decay 1e−2is used. The model is trained with batch size 64 and MAE as the loss function. Early stop is performed by monitoring the validation error.

    def forward(self, feature: Tensor, inputs: Tensor, targets: Tensor = None, batch_seen: int = 0) -> Tensor:
        """
        dynamic convolutional recurrent neural network
        :param feature: [N, d]
        :param inputs: [B, n_hist, N, input_dim]
        :param targets: exists for training, tensor, [B, n_pred, N, output_dim]
        :param batch_seen: int, the number of batches the model has seen
        :return: [B, n_pred, N, output_dim]
        """

        inputs = inputs.view(self.seq_len, -1, self.num_nodes, self.input_dim).permute(1, 2, 0, 3).contiguous().view(-1, self.num_nodes, self.seq_len * self.input_dim)
        # targets = targets.view(self.horizon, -1, self.num_nodes, self.input_dim).permute(1, 0, 2, 3).contiguous()
        #
        # feature = self.geo_encoder(feature.float())
        # states = self.encoder(feature, inputs)

        # targets = None


        # outputs = self.decoder(feature, states, targets, self._compute_sampling_threshold(batch_seen))
        outputs = self.fc2(torch.sigmoid(self.fc1(inputs))) #bnt
        # return outputs.permute(1, 0, 2, 3).contiguous().view(self.horizon, -1, self.num_nodes * self.output_dim)
        return outputs.view(-1, self.num_nodes, self.horizon, self.output_dim).permute(2, 0, 1, 3).contiguous().view(self.horizon, -1, self.num_nodes * self.output_dim)

    def _compute_sampling_threshold(self, batches_seen: int):
        return self.cl_decay_steps / (self.cl_decay_steps + math.exp(batches_seen / self.cl_decay_steps))

        # :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
        #          y: shape (horizon, batch_size, num_sensor * output_dim)

# def test():
#     dist = np.random.randn(207, 207)
#     edge1, edge2 = [[] for _ in range(207)], [[] for _ in range(207)]
#     for i in range(207):
#         for j in range(207):
#             if np.random.random() < 0.2:
#                 edge1[i].append(j)
#                 edge2[j].append(i)
#     me = STMetaEncoder(2, 32, 32, 32, [32, 4], (dist, edge1, edge2), 32)
#     md = STMetaDecoder(12, 1, 32, 32, 32, [32, 4], (dist, edge1, edge2), 32)
#     data = torch.randn(31, 12, 207, 2)
#     feature = torch.randn(207, 32)
#     states = me(feature, data)
#     print(states[0].shape, states[1].shape)
#     outputs = md(feature, states)
# m = STMetaNet((dist, edge1, edge2), 12, 2, 1, 2000, ['NormalGRU', 'MetaGRU'], [], [16, 2], [32, 32])




