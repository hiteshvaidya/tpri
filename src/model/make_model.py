import torch
import torch.nn.functional as torchfunc
from src.model.rnn import RNN
from src.model.gru import GRU
from pipeline.save_load import device


def make_model(train_data, network='rnn', loss='squared',
               reg_param=0., non_lin='tanh',
               dim_output=None, input_size=1, hidden_size=100):
    """
    Given the entries of this function, generates the corresponding rnn and loss
    Args:
        train_data: (torch.utils.data.DataLoader) training data, parameters of which are used to define the network
                such as the size of the inputs
        network: (str) choice between 'rnn' or 'gru
        loss: (str) choice between 'mean_squared' and 'cross_entropy'
        reg_param: (float) regularization parameter for a squared loss
                (not implemented yet so it would throw an error if reg>0)
        non_lin: (str) choice of non-linear acitvation function
        dim_output: (int) dimension of the output (typically number of class for a classification problem)
        input_size: (int) size of the inputs
        hidden_size: (int) size of the hidden states of the recurrent network

    Returns:
        loss, regularization, net: the loss, the regularization and the network defining the problem

    """
    input, target = next(iter(train_data))
    if loss in ['squared', 'mean_squared']:
        dim_output = target.shape[1]
    elif loss in ['cross_entropy']:
        assert dim_output is not None
    else:
        raise NotImplementedError

    if network == 'rnn':
        net = RNN(input_size, hidden_size, dim_output, activ=non_lin)
    elif network == 'gru':
        net = GRU(input_size, hidden_size, dim_output)
    else:
        raise NotImplementedError

    net = net.to(device)

    if loss == 'mean_squared':
        loss = torch.nn.MSELoss()
    elif loss == 'cross_entropy':
        loss = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    if reg_param > 0.:
        regularization = Regularization(reg_param)
        raise NotImplementedError('Target propagation with a regularization on the parameters is not implemented yet')
    else:
        def regularization(x):
            return 0.

    return loss, regularization, net


class Regularization:
    def __init__(self, reg_param):
        self.reg_param = reg_param

    def __call__(self, net):
        return self.reg_param * 0.5 * sum([torch.norm(param) ** 2 for name, param in net.named_parameters()
                                           if 'bias' not in name.split('.')])
