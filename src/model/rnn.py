import numpy as np
import math
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn import init, functional as torchfunc
from torch.nn.parameter import Parameter

from torch.nn.modules import RNN as RNNLoop
from pipeline.save_load import device


class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activ='tanh'):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        if activ == 'tanh':
            self.activ = torch.tanh
        elif activ == 'relu':
            self.activ = torch.relu
        elif activ == 'none':
            self.activ = None
        else:
            raise NotImplementedError

        if activ != 'none':
            self.rnn = RNNLoop(input_size, hidden_size, nonlinearity=activ, batch_first=True)
        else:
            self.rnn = LinearRNNLoop(input_size, hidden_size)
        init_params(self.rnn.weight_ih_l0, self.rnn.bias_ih_l0)
        init_params(self.rnn.weight_hh_l0, self.rnn.bias_hh_l0)

        self.predict = Layer(hidden_size, output_size)
        init_params(self.predict.lin.weight, self.predict.lin.bias)

        self.reverse_Vhh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.reverse_ch = Parameter(torch.Tensor(hidden_size))
        init_params(self.reverse_Vhh, self.reverse_ch)

    def rep(self, inputs):
        batch_size = inputs.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)

        seq_h, h = self.rnn(inputs, h0)
        seq_h = torch.cat((torch.zeros(batch_size, 1, self.hidden_size, device=device), seq_h), dim=1)
        return seq_h, h.squeeze(0)

    def forward(self, inputs):
        return self.predict(self.rep(inputs)[1])

    def target_prop(self, inputs, hiddens, target, diff_mode='linearized', reverse_mode='auto_enc', tol_inv=1e-3):
        out = 0.
        for i in range(inputs.size(1) - 1, -1, -1):
            inpt = inputs[:, i, :]
            add_out = 0.5 * torch.sum((hiddens[:, i+1, :] - target) * (hiddens[:, i+1, :] - target))
            if reverse_mode == 'auto_enc':
                add_out = add_out/inpt.size(0)
            out = out + add_out
            with torch.no_grad():
                if diff_mode == 'diff':
                    target = hiddens[:, i, :] - self.reverse_layer(inpt, hiddens[:, i+1, :], reverse_mode, tol_inv) \
                             + self.reverse_layer(inpt, target, reverse_mode, tol_inv)
                elif diff_mode == 'linearized':
                    # def func(h):
                    #     return torch.sum(self.reverse_layer(inpt, h, reverse_mode, tol_inv))
                    # jac_vec = torch.autograd.functional.jvp(func, hiddens[:, i+1, :], target-hiddens[:, i+1, :])[1]
                    jac_vec = self.jac_reverse_layer(hiddens[:, i+1, :], target-hiddens[:, i+1, :])
                    target = hiddens[:, i, :] + jac_vec
                elif diff_mode == 'none':
                    target = self.reverse_layer(inpt, target, reverse_mode, tol_inv)
                elif diff_mode == 'test':
                    target = hiddens[:, i, :]

        out.backward()

    def reverse_layer(self, input, target, reverse_mode='auto_enc', tol_inv=1e-3):
        if reverse_mode == 'auto_enc':
            target = self.activ(F.linear(target, self.reverse_Vhh, self.reverse_ch) + F.linear(input, self.rnn.weight_ih_l0))
        elif reverse_mode in ['optim', 'optim_variant']:
            if self.activ == torch.tanh:
                target = torch.atanh(torch.clamp(target, -1 + tol_inv, 1 - tol_inv))
            elif self.activ == torch.relu:
                target = torch.relu(target)
            elif self.activ is None:
                pass
            else:
                raise NotImplementedError
            if reverse_mode == 'optim':
                W = self.rnn.weight_hh_l0
            else:
                W = 2*torch.eye(self.rnn.weight_hh_l0.shape[0], device=device) - self.rnn.weight_hh_l0
            target = torch.cholesky_solve(F.linear(target - F.linear(input, self.rnn.weight_ih_l0,
                                                                     self.rnn.bias_ih_l0 + self.rnn.bias_hh_l0),
                                                   W.t()).t(), self.inv_layer).t()
        return target

    def jac_reverse_layer(self, hidden, dir, tol_inv=1e-3):
        if self.activ == torch.tanh:
            dir = dir/(1-torch.clamp(hidden, -1 + tol_inv, 1 - tol_inv)**2)
            out = torch.cholesky_solve(F.linear(dir, self.rnn.weight_hh_l0.t()).t(), self.inv_layer).t()
        else:
            raise NotImplementedError
        return out

    def update_reverse(self, inputs=None, hiddens=None, reverse_mode='auto_enc', sigma_noise=0., reg=0.):
        if reverse_mode == 'auto_enc':
            out = 0.
            for i in range(inputs.size(1)):
                hidden = deepcopy(hiddens[:, i, :].data)
                inpt = inputs[:, i, :]
                noise = sigma_noise * torch.randn(hidden.shape, device=device)

                forward_step = self.rnn(inpt.unsqueeze(1), (hidden + noise).unsqueeze(0))[1].squeeze()
                for_back_step = self.reverse_layer(inpt, forward_step)
                out = out + 0.5 * torch.sum((for_back_step - hidden - noise) * (for_back_step - hidden - noise))/inpt.size(0)

            grad_Vhh, grad_ch = torch.autograd.grad(out, [self.reverse_Vhh, self.reverse_ch])
            self.reverse_Vhh.grad = grad_Vhh
            self.reverse_ch.grad = grad_ch
        elif reverse_mode in ['optim', 'optim_variant']:
            if reverse_mode == 'optim':
                W = self.rnn.weight_hh_l0
            else:
                W = 2*torch.eye(self.rnn.weight_hh_l0.shape[0], device=device) - self.rnn.weight_hh_l0
                reg = 0
            try:
                self.inv_layer = torch.cholesky(W.t().mm(W)
                                                + reg * torch.eye(W.shape[0], device=device))
            except:
                self.inv_layer = None
        else:
            raise NotImplementedError


def init_params(weight, bias=None, init_type='orth'):
    if init_type == 'orth':
        weight.data = rand_orth(weight.shape, np.sqrt(6. / sum(weight.shape)))
        if bias is not None:
            bias.data = torch.zeros_like(bias)
    elif init_type == 'default':
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(torch.zeros())
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)
    else:
        raise NotImplementedError


def rand_orth(shape, irange):
    A = - irange + 2 * irange * torch.rand(*shape)
    U, s, V = torch.svd(A)
    return torch.mm(U, torch.mm(torch.eye(U.shape[1], V.shape[1]), V.t()))


class LinearRNNLoop(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True, activ=None):
        super(LinearRNNLoop, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.weight_ih_l0 = Parameter(torch.Tensor(hidden_size, input_size))
        self.bias_ih_l0 = Parameter(torch.Tensor(hidden_size))
        self.weight_hh_l0 = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_hh_l0 = Parameter(torch.Tensor(hidden_size))
        self.activ = activ

    def forward(self, inputs, hidden):
        hidden = hidden[0]
        assert self.batch_first
        seq_h = []
        for i in range(inputs.shape[1]):
            hidden = F.linear(inputs[:, i, :], self.weight_ih_l0, self.bias_ih_l0) \
                     + F.linear(hidden, self.weight_hh_l0, self.bias_hh_l0)
            if self.activ is not None:
                hidden = self.activ(hidden)
            seq_h.append(hidden)
        return torch.stack(seq_h).transpose(0, 1), hidden


activs = dict(ReLU=torch.relu, tanh=torch.tanh, sigmoid=torch.sigmoid, softmax=lambda x: torch.softmax(x, dim=1))


class Layer(nn.Module):
    def __init__(self, dim_in, dim_out, activ=None, lin='full', bias=True,
                 pooling=None, pooling_kernel=None, pooling_stride=None, pooling_padding=None,
                 padding=None, stride=None, kernel_size=None, to_vect=False):
        super(Layer, self).__init__()

        self.to_vect = to_vect
        self.activ = activs[activ] if activ is not None else None

        if lin == 'full':
            self.lin = nn.Linear(dim_in, dim_out, bias=bias)
        elif lin == 'conv':
            self.lin = nn.Conv2d(dim_in, dim_out, kernel_size, padding=padding, stride=stride, bias=bias)
        else:
            raise NotImplementedError

        if pooling == 'avg':
            self.pooling = nn.AvgPool2d(kernel_size=pooling_kernel, stride=pooling_stride, padding=pooling_padding)
        elif pooling == 'max':
            self.pooling = max_pooling
        else:
            self.pooling = None

    def forward(self, input):
        out = self.lin(input)
        if self.activ is not None:
            out = self.activ(out)
        if self.pooling is not None:
            out = self.pooling(out)
        if self.to_vect:
            out = out.view(out.shape[0], -1)
        return out


def max_pooling(out):
    out = torch.mean(torch.reshape(out, (out.shape[0], out.shape[1], -1)), 2)
    return torchfunc.softmax(out, dim=1)



