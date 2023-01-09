import numpy as np
import math
import time
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules import RNN as RNNLoop
from pipeline.save_load import device

from src.model.rnn import init_params, rand_orth, Layer


class RNNText(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size=10000, activ='tanh', freeze_embedding=False):
        super(RNNText, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = embedding_dim

        if activ == 'tanh':
            self.activ = torch.tanh
        elif activ == 'relu':
            self.activ = torch.relu
        else:
            raise NotImplementedError

        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        if freeze_embedding:
            self.embedding._parameters['weight'].requires_grad = False
        self.rnn = RNNLoop(embedding_dim, hidden_size, nonlinearity=activ, batch_first=True)
        init_params(self.rnn.weight_ih_l0, self.rnn.bias_ih_l0)
        init_params(self.rnn.weight_hh_l0, self.rnn.bias_hh_l0)

        self.predict = Layer(hidden_size, vocab_size)
        init_params(self.predict.lin.weight, self.predict.lin.bias)

        self.reverse_Vhh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.reverse_ch = Parameter(torch.Tensor(hidden_size))
        init_params(self.reverse_Vhh, self.reverse_ch)

    def rep(self, inputs):
        # Could use stateful implementation with careful treatment of batch feeding
        # torchtext could be used in that purpose
        # see https://stackoverflow.com/questions/58276337/proper-way-to-feed-time-series-data-to-stateful-lstm for
        # more info
        batch_size = inputs.size(0)
        hidden = torch.zeros(1, batch_size, self.hidden_size, device=device, requires_grad=True)
        embedded = self.embedding(inputs)
        seq_h, h = self.rnn(embedded, hidden)
        seq_h = torch.cat((torch.zeros(batch_size, 1, self.hidden_size, device=device), seq_h), dim=1)
        return seq_h, h.squeeze(0)

    def forward(self, inputs):
        # trranspose needed otherwise not contiguous dimensions to be reshaped
        # seq_h = self.rep(inputs)[0].transpose(0, 1)
        seq_h = self.rep(inputs)[0][:, 1:]
        pred = self.predict(seq_h.reshape(seq_h.shape[0]*seq_h.shape[1], seq_h.shape[2]))
        pred = pred.view(seq_h.shape[0], seq_h.shape[1], pred.shape[1])
        return pred

    def target_prop(self, inputs, hiddens, targets, diff_mode='linearized', reverse_mode='auto_enc', tol_inv=1e-3):
        out = 0.
        target = targets[:, -1, :]
        for i in range(inputs.size(1) - 1, -1, -1):
            inpt = self.embedding(inputs[:, i])
            add_out = 0.5 * torch.sum((hiddens[:, i+1, :] - target) * (hiddens[:, i+1, :] - target))
            if reverse_mode == 'auto_enc':
                add_out = add_out/inpt.size(0)
            out = out + add_out
            with torch.no_grad():
                if i > 0:
                    if diff_mode == 'diff':
                        target = hiddens[:, i, :] \
                                 - 2*self.reverse_layer(inpt, hiddens[:, i+1, :], reverse_mode, tol_inv) \
                                 + self.reverse_layer(inpt, target, reverse_mode, tol_inv) \
                                 + self.reverse_layer(inpt, targets[:, i, :], reverse_mode, tol_inv)
                    elif diff_mode == 'linearized':
                        jac_vec = self.jac_reverse_layer(hiddens[:, i+1, :],
                                                         target + targets[:, i, :] - 2*hiddens[:, i+1, :])
                        target = hiddens[:, i, :] + jac_vec
                    else:
                        target = self.reverse_layer(inpt, target, reverse_mode, tol_inv)\
                                 + self.reverse_layer(inpt, targets[:, i, :], reverse_mode, tol_inv)
        out.backward()

    def reverse_layer(self, input, target, reverse_mode='auto_enc', tol_inv=1e-3):
        if reverse_mode == 'auto_enc':
            target = self.activ(F.linear(target, self.reverse_Vhh, self.reverse_ch) + F.linear(input, self.rnn.weight_ih_l0))
        else:
            if self.activ == torch.tanh:
                target = torch.atanh(torch.clamp(target, -1 + tol_inv, 1 - tol_inv))
            elif self.activ == torch.relu:
                target = torch.relu(target)
            else:
                raise NotImplementedError
            target = torch.cholesky_solve(F.linear(target - F.linear(input, self.rnn.weight_ih_l0,
                                                                     self.rnn.bias_ih_l0 + self.rnn.bias_hh_l0),
                                                   self.rnn.weight_hh_l0.t()).t(), self.inv_layer).t()
        return target

    def jac_reverse_layer(self, hidden, dir, tol_inv=1e-3):
        if self.activ == torch.tanh:
            dir = dir/(1-torch.clamp(hidden, -1 + tol_inv, 1 - tol_inv)**2)
            out = torch.cholesky_solve(F.linear(dir, self.rnn.weight_hh_l0.t()).t(), self.inv_layer).t()
        else:
            raise NotImplementedError
        return out

    def update_reverse(self, inputs=None, hiddens=None, reverse_mode='auto_enc', sigma_noise=0., reg=0., mse=False):
        if reverse_mode == 'auto_enc':
            out = 0.
            for i in range(inputs.size(1)):
                hidden = deepcopy(hiddens[:, i, :].data)
                inpt = inputs[:, i].unsqueeze(-1) if self.input_size == 1 else inputs[:, i, :]
                noise = sigma_noise * torch.randn(hidden.shape, device=device)

                forward_step = self.rnn(inpt.unsqueeze(1), (hidden + noise).unsqueeze(0))[1].squeeze()
                for_back_step = self.reverse_layer(inpt, forward_step)
                out = out + 0.5 * torch.sum((for_back_step - hidden - noise) * (for_back_step - hidden - noise))/inpt.size(0)

            grad_Vhh, grad_ch = torch.autograd.grad(out, [self.reverse_Vhh, self.reverse_ch])
            self.reverse_Vhh.grad = grad_Vhh
            self.reverse_ch.grad = grad_ch
        else:
            self.inv_layer = torch.cholesky(self.rnn.weight_hh_l0.t().mm(self.rnn.weight_hh_l0)
                                            + reg * torch.eye(self.rnn.weight_hh_l0.shape[0], device=device))

