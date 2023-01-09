import torch
from copy import deepcopy
from torch.nn.modules import GRU as GRULoop
from pipeline.save_load import device
import torch.nn.functional as F
from src.model.rnn import Layer
from src.model.rnn import init_params
from src.optim.inner_algos import InnerBarzilaiBorwein


class GRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Recall the architecture
        r_t = sigmoid(W_{ir} x_t + W_{hr} h_{t-1} + b_{ir} + b_{hr}) \\
		z_t = sigmoid(W_{iz} x_t + W_{hz} h_{t-1} + b_{iz} + b_{hz}) \\
		n_t = tanh(W_{in} x_t + b_{in} + r_t odot (W_{hn} h_{t-1} + b_{hn})) \\
		h_t = (1 - z_t) odot h_{t-1} + z_t odot n_t

		Weights are concatenated in the order r, z, n for the W_{h_}, W_{i_} and the biases
        """

        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.embedding = GRULoop(input_size, hidden_size, batch_first=True)
        for i in range(3):
            init_params(self.embedding.weight_ih_l0[hidden_size*i:hidden_size*(i+1)],
                        self.embedding.bias_ih_l0[hidden_size*i:hidden_size*(i+1)])
            init_params(self.embedding.weight_hh_l0[hidden_size*i:hidden_size*(i+1)],
                        self.embedding.bias_hh_l0[hidden_size*i:hidden_size*(i+1)])
        self.predict = Layer(hidden_size, output_size)
        init_params(self.predict.lin.weight, self.predict.lin.bias)
        # Inv layers are listed in the same order as the weihts
        self.inv_layer = [None for i in range(3)]

    def rep(self, inputs):
        batch_size = inputs.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)

        seq_h, h = self.embedding(inputs, h0)
        seq_h = torch.cat((torch.zeros(batch_size, 1, self.hidden_size, device=device), seq_h), dim=1)
        return seq_h, h.squeeze(0)

    def forward(self, inputs):
        return self.predict(self.rep(inputs)[1])

    def target_prop(self, inputs, hiddens, target, diff_mode='diff', reverse_mode='auto_enc', tol_inv=None, mse=False):
        out = 0.
        for i in range(inputs.size(1) - 1, -1, -1):
            inpt = inputs[:, i, :]
            add_out = 0.5 * torch.sum((hiddens[:, i+1, :] - target) * (hiddens[:, i+1, :] - target))
            out = out + add_out
            if reverse_mode != 'optim':
                if diff_mode == 'diff':
                    target = hiddens[:, i, :] - self.reverse_layer(inpt, hiddens[:, i + 1, :], reverse_mode, tol_inv, hiddens[:, i, :]) \
                             + self.reverse_layer(inpt, target, reverse_mode, tol_inv, hiddens[:, i, :])
                elif diff_mode == 'linearized':
                    def func(h):
                        return torch.sum(self.reverse_layer(inpt, h, reverse_mode, tol_inv, hiddens[:, i, :]))

                    jac_vec = torch.autograd.functional.jvp(func, hiddens[:, i + 1, :], target - hiddens[:, i + 1, :])[1]
                    target = hiddens[:, i, :] + jac_vec
                elif diff_mode == 'none':
                    target = self.reverse_layer(inpt, target, reverse_mode, tol_inv)
            else:
                with torch.no_grad():
                    prev_dir = target - hiddens[:, i + 1, :]
                    dimh = self.hidden_size
                    # Recompute the intermediate quantities of the forward pass
                    hidden = hiddens[:, i, :]
                    r = torch.sigmoid(F.linear(hidden, self.embedding.weight_hh_l0[:dimh], self.embedding.bias_hh_l0[:dimh]) +
                                      F.linear(inpt, self.embedding.weight_ih_l0[:dimh], self.embedding.bias_ih_l0[:dimh]))
                    z = torch.sigmoid(F.linear(hidden, self.embedding.weight_hh_l0[dimh:2*dimh], self.embedding.bias_hh_l0[dimh:2*dimh]) +
                                      F.linear(inpt, self.embedding.weight_ih_l0[dimh:2*dimh], self.embedding.bias_ih_l0[dimh:2*dimh]))
                    a = F.linear(hidden, self.embedding.weight_hh_l0[2*dimh:3*dimh], self.embedding.bias_hh_l0[2*dimh:3*dimh])
                    b = F.linear(inpt, self.embedding.weight_ih_l0[2*dimh:3*dimh], self.embedding.bias_ih_l0[2*dimh:3*dimh]) + a*r
                    n = torch.tanh(b)

                    # Compute the next direction, see draft
                    dir = (1-z)*prev_dir
                    auxz = torch.clamp(z, 0 + tol_inv, 1 - tol_inv)
                    auxz = 1/(auxz*(1-auxz))
                    auxz = auxz*(n-hidden)*prev_dir
                    Wh = self.embedding.weight_hh_l0[dimh:2*dimh]
                    dir = dir + torch.cholesky_solve(F.linear(auxz, Wh.t()).t(), self.inv_layer[1]).t()
                    auxr = torch.clamp(r, 0 + tol_inv, 1 - tol_inv)
                    auxr = 1 / (auxr * (1 - auxr))
                    aux0 = (1 - torch.tanh(b)**2) * z * prev_dir
                    auxr = auxr * a * aux0
                    Wh = self.embedding.weight_hh_l0[:dimh]
                    dir = dir + torch.cholesky_solve(F.linear(auxr, Wh.t()).t(), self.inv_layer[0]).t()
                    aux1 = r*aux0
                    Wh = self.embedding.weight_hh_l0[2*dimh:3*dimh]
                    dir = dir + torch.cholesky_solve(F.linear(aux1, Wh.t()).t(), self.inv_layer[2]).t()

                    target = hidden + dir
        out.backward()

    def reverse_layer(self, input, target, reverse_mode='optim', tol_inv=1e-6, hidden=None, reg=1., operation='none'):
        if reverse_mode == 'optim_algo':
            def forward(h):
                return self.embedding(input.unsqueeze(1), h.unsqueeze(0))[1].squeeze()

            def func(h):
                return 0.5*torch.sum((forward(h)-target)*(forward(h)-target)) + 0.5*reg*torch.sum(h)**2

            optimizer = InnerBarzilaiBorwein(max_iter=5)
            hidden = deepcopy(hidden.data)
            hidden.requires_grad = True
            out = optimizer.solve(func, hidden).data
        elif reverse_mode == 'optim':
            pass

        else:
            raise NotImplementedError
        return out

    def update_reverse(self, inputs=None, hiddens=None, reverse_mode='auto_enc', sigma_noise=0., reg=0., mse=False):
        for i in range(3):
            # try:
            dimh = self.hidden_size
            W = self.embedding.weight_hh_l0[i*dimh:(i+1)*dimh]
            self.inv_layer[i] = torch.cholesky(W.t().mm(W) + reg * torch.eye(W.shape[0], device=device))
            # except:
            #     self.inv_layer = None

