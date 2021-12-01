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
    """
    RNN as in Pytorch, except that TP with regularized inverses is implemented as part of the module
    """
    def __init__(self, input_size, hidden_size, output_size, activ='tanh'):
        """
        Args:
            input_size: (int) size of the input of the RNN at each time step
            hidden_size: (int) size of the hidden state of the RNN
            output_size: (int) size of the output
            activ: (str) choice of nonlinear activation function
        """
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

        # RNNLoop is the usual loop of Pytorch that benefits from optimized implementation on GPUs
        if activ != 'none':
            self.rnn = RNNLoop(input_size, hidden_size, nonlinearity=activ, batch_first=True)
        else:
            self.rnn = LinearRNNLoop(input_size, hidden_size)

        # Initialize the parameters of the transition operations
        init_params(self.rnn.weight_ih_l0, self.rnn.bias_ih_l0)
        init_params(self.rnn.weight_hh_l0, self.rnn.bias_hh_l0)

        # Last layer is just a linear one with the appropriate sizes
        self.predict = nn.Linear(hidden_size, output_size, bias=True)
        init_params(self.predict.lin.weight, self.predict.lin.bias)

        # Initialize parameters of reversed layer if one wants to implement TP with parameterized layers
        self.reverse_Vhh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.reverse_ch = Parameter(torch.Tensor(hidden_size))
        init_params(self.reverse_Vhh, self.reverse_ch)

    def rep(self, inputs):
        """
            Take a sequence of inputs and outputs the corresponding sequence of hidden states
            and the last computed hidden state
            Args:
                inputs: (torch.Tensor) of shape (batch_size, length_seq, input_size)
                    Sequence of inputs fed to the RNN

            Returns:
                 seq_h: (torch.Tensor) of shape (batch_size, length_seq, hidden_size)
                    Sequence of hidden states computed by the RNN
                h: (torch.Tensor) of shape (batch_size, hidden_size)
                    Final ouput of the RNN (before feeding it to the prediction layer)
        """
        batch_size = inputs.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)

        seq_h, h = self.rnn(inputs, h0)
        seq_h = torch.cat((torch.zeros(batch_size, 1, self.hidden_size, device=device), seq_h), dim=1)
        return seq_h, h.squeeze(0)

    def forward(self, inputs):
        """
        Computes the last hidden state corresponding to the inputs and feed it to the prediction layer
        Args:
            inputs: (torch.Tensor) of shape (batch_size, length_seq, input_size)
                    Sequence of inputs fed to the RNN
        Returns:
            output: (torch.Tensor) of shape (batch_size, output_size)
        """
        return self.predict(self.rep(inputs)[1])

    def target_prop(self, inputs, hiddens, target, diff_mode='linearized', reverse_mode='auto_enc', tol_inv=1e-3):
        """
        Back-propagates targets
        Args:
            inputs: (torch.Tensor) of shape (batch_size, length_seq, input_size)
                    Sequence of inputs fed to the RNN
            hiddens: (torch.Tensor) of shape (batch_size, length_seq, hidden_size)
                    Sequence of hidden states computed by the network
            target: (torch.Tensor) of shape (batch_size, hidden_size)
                    Initial target computed as a gradient step on the prediction layer
            diff_mode (str) method to back-propagate targets
                    if 'diff', uses the difference target propagation formula, i.e.,
                        prev_target = prev_hidden_state + inv_layer(target) - inv_layer(hidden_state)
                        where prev_target means the target at time t-1, while target is the target at time t
                    if 'linearized', uses the linearized target propagation formula
                    (i.e. the Jacobian of the regularized inverse)
                    if 'none', uses directly the appinverse
            reverse_mode (str): what approximation of the inverse is used
                    if 'auto_enc', uses an additional layer to approximate the inverse
                    if 'optim', uses a regualrized inverse
            tol_inv: (float) parameter used to stabilize the computation of the inverses of e.g. tanh
        Returns:
            Nothing, the gradients are automatically stored in the parameters as in a classical backward operation
        """
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
                    jac_vec = self.jac_reverse_layer(hiddens[:, i+1, :], target-hiddens[:, i+1, :])
                    target = hiddens[:, i, :] + jac_vec
                elif diff_mode == 'none':
                    target = self.reverse_layer(inpt, target, reverse_mode, tol_inv)
        out.backward()

    def reverse_layer(self, input, target, reverse_mode='auto_enc', tol_inv=1e-3):
        """
        Reverse layer used in target prop
        Args:
            input: (torch.Tensor) of shape (batch_size, input_size)
                    Current input
            target: (torch.Tensor) of shape (batch_size, hidden_size)
                    Target to be inverted
            reverse_mode (str): what approximation of the inverse is used
                    if 'auto_enc', uses an additional layer to approximate the inverse
                    if 'optim', uses a regularized inverse
            tol_inv: (float) parameter used to stabilize the computation of the inverses of e.g. tanh
        Returns:
            target: (torch.Tensor) of shape (batch_size, hidden_size)
                    approximate inverse of the target
        """
        if reverse_mode == 'auto_enc':
            target = self.activ(F.linear(target, self.reverse_Vhh, self.reverse_ch) + F.linear(input, self.rnn.weight_ih_l0))
        elif reverse_mode in 'optim':
            if self.activ == torch.tanh:
                target = torch.atanh(torch.clamp(target, -1 + tol_inv, 1 - tol_inv))
            elif self.activ == torch.relu:
                target = torch.relu(target)
            elif self.activ is None:
                pass
            else:
                raise NotImplementedError
            W = self.rnn.weight_hh_l0
            target = torch.cholesky_solve(F.linear(target - F.linear(input, self.rnn.weight_ih_l0,
                                                                     self.rnn.bias_ih_l0 + self.rnn.bias_hh_l0),
                                                   W.t()).t(), self.inv_layer).t()
        return target

    def jac_reverse_layer(self, hidden, dir, tol_inv=1e-3):
        """
        Jacobian of the reverse layer used in target prop
        Args:
            hidden: (torch.Tensor) of shape (batch_size, hidden_size)
                    Current hidden state
            dir: (torch.Tensor) of shape (batch_size, hidden_size)
                    direction (target - hidden) on which to apply the jacobian of the reverse layer
            tol_inv: (float) parameter used to stabilize the computation of the inverses of e.g. tanh
        Returns:
            new_dir: (torch.Tensor) of shape (batch_size, hidden_size)
                    output after applying the Jacobian of the regularized inverse
        """
        if self.activ == torch.tanh:
            dir = dir/(1-torch.clamp(hidden, -1 + tol_inv, 1 - tol_inv)**2)
            new_dir = torch.cholesky_solve(F.linear(dir, self.rnn.weight_hh_l0.t()).t(), self.inv_layer).t()
        else:
            raise NotImplementedError
        return new_dir

    def update_reverse(self, inputs=None, hiddens=None, reverse_mode='auto_enc', sigma_noise=0., reg=0.):
        """
        Compute an approximate inverse of the layer of the RNN
        Args:
            inputs: (torch.Tensor) of shape (batch_size, length_seq, input_size)
                    Sequence of inputs fed to the RNN
            hiddens: (torch.Tensor) of shape (batch_size, length_seq, hidden_size)
                    Sequence of hidden states computed by the network
            reverse_mode (str): what approximation of the inverse is used
                    if 'auto_enc', updates a reverse layer by a gradient step
                        on the squared loss between the inputs and the outputs
                    if 'optim', computes a regularized inverse
            sigma_noise: (float) additional noise added to compute a gradient step
                        on the reverse layer for the auto_enc mode
            reg: (float) regularization for the regularized inverse
        Returns:
            Nothing, just keep in memory the inverse layer to be used in the target propagation

        """
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
            W = self.rnn.weight_hh_l0
            try:
                self.inv_layer = torch.cholesky(W.t().mm(W)
                                                + reg * torch.eye(W.shape[0], device=device))
            except:
                self.inv_layer = None
        else:
            raise NotImplementedError


def init_params(weight, bias=None, init_type='orth'):
    """
    Initialize the weights of the network using orthonormal weights.
    """
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
    """
    Compute a random orthonormal matrix
    """
    A = - irange + 2 * irange * torch.rand(*shape)
    U, s, V = torch.svd(A)
    return torch.mm(U, torch.mm(torch.eye(U.shape[1], V.shape[1]), V.t()))


class LinearRNNLoop(torch.nn.Module):
    """
    Defines a RNN loop without activation functions

    """
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

