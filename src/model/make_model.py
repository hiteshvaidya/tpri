import torch
import torch.nn.functional as torchfunc
from src.model.rnn import RNN
from src.model.rnn_text import RNNText
from src.model.gru import GRU
from pipeline.save_load import device


def make_model(train_data, network='mlp', loss='squared',
               reg_param=0., non_lin='tanh',
               dim_output=None, input_size=1, hidden_size=100,
               embedding_dim=1024, freeze_embedding=False):
    input, target = next(iter(train_data))
    if loss in ['squared', 'mean_squared']:
        dim_output = target.shape[1]
    elif loss in ['cross_entropy', 'cross_entropy_seq']:
        assert dim_output is not None
    else:
        raise NotImplementedError

    if network == 'rnn':
        net = RNN(input_size, hidden_size, dim_output, activ=non_lin)
    elif network == 'gru':
        net = GRU(input_size, hidden_size, dim_output)
    elif network == 'rnn_text':
        net = RNNText(embedding_dim, hidden_size, vocab_size=dim_output, activ=non_lin,
                      freeze_embedding=freeze_embedding)
    else:
        raise NotImplementedError

    net = net.to(device)

    if loss == 'mean_squared':
        loss = torch.nn.MSELoss()
    elif loss == 'cross_entropy':
        loss = torch.nn.CrossEntropyLoss()
    elif loss == 'cross_entropy_seq':
        loss = CrossEntropySeq()
    else:
        raise NotImplementedError

    if reg_param > 0.:
        regularization = Regularization(reg_param)
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


class CrossEntropySeq:
    def __init__(self):
        pass

    def __call__(self, pred, truth):
        return torchfunc.cross_entropy(pred.view(-1, pred.shape[-1]), truth.view(-1))
