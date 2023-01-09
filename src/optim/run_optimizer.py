import math
import torch
import time
from copy import deepcopy
import torch.nn.functional as F

from src.optim.custom_algos import CustomSGD, CustomAdam
from pipeline.save_load import device


def run_optim(loss, regularization, net, train_data, test_data, nb_iter_per_log,
              oracle='grad', lr=0.1, max_iter=100,
              momentum=0., reg=0., diff_mode='linearized',
              algo='sgd', tol_inv=1e-3,
              lr_reverse=0.1, noise=0.,
              lr_target=0.1, log_target_norms=False,
              log_grad_norms=False, log_spec_rad=False,
              input=None, aux_vars=None, log=None,
              verbose=True, logging=True, averaging=0):
    saved_params = []
    aux_net = deepcopy(net) if averaging > 0 else None
    nesterov = True if momentum != 0. else False
    grad_norms = torch.tensor(0.) if log_grad_norms else None
    if oracle == 'target_auto_enc':
        forward_params = [param for name, param in net.named_parameters() if 'reverse' not in name]
        backward_params = [param for name, param in net.named_parameters() if 'reverse' in name]
        optimizer = CustomSGD([{'params': forward_params},
                               {'params': backward_params, 'lr': lr_reverse}],
                              lr=lr, momentum=momentum, nesterov=nesterov)
    else:
        if algo == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
        elif algo == 'adam':
            optimizer = CustomAdam(net.parameters(), lr=lr)
        elif algo == 'adagrad':
            optimizer = torch.optim.Adagrad(net.parameters(), lr=lr)
        else:
            raise NotImplementedError

    if input is not None:
        net.load_state_dict(input)
        if averaging == 0:
            optimizer.load_state_dict(aux_vars)
        else:
            optimizer.load_state_dict(aux_vars[0])
            saved_params = aux_vars[1]

    if log is None:
        init_iter = 0
        saved_params.append([params for params in net.parameters()])
        if log_target_norms and 'target' in oracle:
            net.update_reverse(reg=reg, reverse_mode=oracle.replace('target_', ''))
        log = collect_info(log, loss, regularization, net, aux_vars, train_data, test_data, init_iter, verbose, logging,
                           diff_mode='linearized', oracle=oracle, tol_inv=tol_inv, log_target_norms=log_target_norms,
                           grad_norms=grad_norms, log_spec_rad=log_spec_rad,
                           averaging=averaging, saved_params=saved_params, aux_net=aux_net)
    else:
        init_iter = log['iteration'][-1]
    iteration = init_iter+1
    assess_time = True
    start_time = time.time()
    while iteration <= max_iter:
        optimizer.zero_grad()
        for input, label in train_data:
            if iteration > max_iter:
                break
            optimizer.zero_grad()
            # input = input.type(torch.get_default_dtype()).to(device)
            input = input.to(device)
            label = label.to(device)
            if oracle == 'grad':
                out = loss(net(input), label) + regularization(net)
                out.backward()
                if log_grad_norms:
                    grad_norms = sum([torch.norm(param.grad)
                                     for name, param in net.rnn.named_parameters()])
                optimizer.step()
            elif 'target' in oracle:
                reverse_mode = oracle.replace('target_', '')
                states, output = net.rep(input)
                if reverse_mode == 'auto_enc':
                    net.update_reverse(input, states, sigma_noise=noise, reverse_mode=reverse_mode)
                    optimizer.step(1)
                elif reverse_mode in ['optim', 'optim_variant']:
                    net.update_reverse(reg=reg, reverse_mode=reverse_mode)
                    if hasattr(net, 'inv_layer') and net.inv_layer is None:
                        collect_info(log, loss, regularization, net, aux_vars, train_data, test_data, iteration,
                                     verbose,
                                     diff_mode='linearized', oracle=oracle, tol_inv=tol_inv,
                                     log_target_norms=log_target_norms,
                                     grad_norms=grad_norms, log_spec_rad=log_spec_rad,
                                     logging=logging, averaging=averaging, aux_net=aux_net)
                        print('Fail inverse')
                        log['train_loss'][-1] = float('nan')
                        break
                if type(loss).__name__ in ['CrossEntropyLoss', 'MSELoss']:
                    last_state = deepcopy(output.data)
                    last_state.requires_grad = True
                    out = loss(net.predict(last_state), label)
                    out.backward()
                    target = last_state - lr_target * last_state.grad
                    net.target_prop(input, states, target, reverse_mode=reverse_mode, diff_mode=diff_mode,
                                    tol_inv=tol_inv)
                elif type(loss).__name__ == 'CrossEntropySeq':
                    # seq_h = deepcopy(states.data).transpose(0, 1)
                    seq_h = deepcopy(states.data[:, 1:])
                    seq_h.requires_grad = True
                    pred = net.predict(seq_h.reshape(seq_h.shape[0] * seq_h.shape[1], seq_h.shape[2]))
                    pred = pred.view(seq_h.shape[0], seq_h.shape[1], pred.shape[1])
                    out = loss(pred, label)
                    out.backward()
                    targets = seq_h - lr_target * seq_h.grad
                    net.target_prop(input, states, targets, reverse_mode=reverse_mode,
                                    diff_mode=diff_mode, tol_inv=tol_inv)

                if log_grad_norms:
                    grad_norms = sum([torch.norm(param.grad)
                                     for name, param in net.rnn.named_parameters()])

                if reverse_mode == 'auto_enc':
                    optimizer.step(0)
                else:
                    optimizer.step()
            if averaging > 0:
                if len(saved_params) > averaging:
                    saved_params.pop(0)
                saved_params.append([params for params in net.parameters()])
            if iteration % nb_iter_per_log == 0:
                collect_info(log, loss, regularization, net, aux_vars, train_data, test_data, iteration, verbose,
                             diff_mode='linearized', oracle=oracle, tol_inv=tol_inv, log_target_norms=log_target_norms,
                             grad_norms=grad_norms, log_spec_rad=log_spec_rad,
                             logging=logging, averaging=averaging, saved_params=saved_params, aux_net=aux_net)
                if assess_time:
                    print('Time for {0} iter: {1}s'.format(nb_iter_per_log, time.time() - start_time))
                    assess_time = False
            iteration += 1

        stopped = check_stop(log)
        if stopped is not None:
            print('{0} {1}'.format(optimizer.__class__.__name__, stopped))
            break

    if max_iter % nb_iter_per_log != 0:
        collect_info(log, loss, regularization, net, aux_vars, train_data, test_data, max_iter, verbose,
                     diff_mode='linearized', oracle=oracle, tol_inv=tol_inv, log_target_norms=log_target_norms,
                     grad_norms=grad_norms, log_spec_rad=log_spec_rad,
                     logging=logging, averaging=averaging, saved_params=saved_params)

    output = net.state_dict()
    if averaging == 0:
        aux_vars = optimizer.state_dict()
    else:
        aux_vars = [optimizer.state_dict(), saved_params]

    return output, aux_vars, log


def collect_info(log, loss, regularization, net, aux_vars, train_data, test_data, iteration, verbose, logging,
                 diff_mode='linearized', oracle='grad', tol_inv=1e-3,
                 log_target_norms=False, grad_norms=None, log_spec_rad=False,
                 averaging=0, saved_params=None, aux_net=None):
    log_header = 'Iter\t\t train loss '
    log_format = '{:0.2f} \t\t {:0.6f} '
    if averaging > 0:
        assert saved_params is not None and aux_net is not None
        avg_params = []
        for i in range(len(saved_params[0])):
            avg_params.append(sum([saved_param[i] for saved_param in saved_params])/len(saved_params))
        for i, param in enumerate(aux_net.parameters()):
            param.data.copy_(avg_params[i].data)
    else:
        aux_net = net
    train_loss = 0.
    for input, target in train_data:
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            train_loss = train_loss + loss(aux_net(input), target) + regularization(aux_net)

    info = dict(iteration=iteration, train_loss=train_loss.item())

    if test_data is not None:
        log_header += '\t\t test loss '
        log_format += '\t\t {:0.6f} '
        log_header += '\t\t accuracy '
        log_format += '\t\t {:0.6f} '
        total = 0
        correct = 0
        test_loss = torch.tensor(0.)
        for input, target in test_data:
            input = input.to(device)
            target = target.to(device)
            with torch.no_grad():
                out = aux_net(input)
                test_loss = test_loss + loss(out, target)
                if type(loss).__name__ == 'CrossEntropyLoss':
                    _, predicted = torch.max(out.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                elif type(loss).__name__ == 'MSELoss':
                    correct += (torch.abs(out-target)<0.04).sum().item()
                    total += target.size(0)
                elif type(loss).__name__ == 'CrossEntropySeq':
                    out = out.view(-1, out.shape[-1])
                    target = target.view(-1)
                    _, predicted = torch.max(out, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                else:
                    raise NotImplementedError

        info.update(test_loss=test_loss.item())
        info.update(accuracy=100 * correct / total)

    if verbose:
        if iteration == 0:
            print(log_header)
        print(log_format.format(*list(info.values())))

    if log_target_norms:
        inputs, label = next(iter(train_data))
        l2_norms = compute_norm_targ(inputs, label, net, loss, oracle=oracle, diff_mode=diff_mode, tol_inv=tol_inv)
        info.update(l2_norms=l2_norms)

    if grad_norms is not None:
        info.update(grad_norms=grad_norms.item())
    if log_spec_rad is not None:
        with torch.no_grad():
            spec_rad = torch.linalg.norm(net.rnn.weight_hh_l0, ord=2)
        info.update(spec_rad=spec_rad.item())

    if logging:
        if log is None:
            log = {key: [info[key]] for key in info.keys()}
        else:
            for key in info.keys():
                log[key].append(info[key])
    return log


def check_stop(log):
    if log['train_loss'][-1] > 2 * log['train_loss'][0] or math.isnan(log['train_loss'][-1]):
        print(log['train_loss'][-1])
        stopped = 'has diverged'
    else:
        stopped = None
    return stopped


def compute_norm_targ(inputs, label, net, loss, oracle='tp', diff_mode='linearized', tol_inv=1e-3):
    inputs = inputs.to(device)
    label = label.to(device)
    hiddens, output = net.rep(inputs)
    last_state = deepcopy(output.data)
    last_state.requires_grad = True
    out = loss(net.predict(last_state), label)
    out.backward()
    target = last_state - last_state.grad
    displacement = last_state.grad

    l2_norms = [torch.norm(displacement).item()]
    for i in range(inputs.size(1) - 1, -1, -1):
        inpt = inputs[:, i, :]
        if 'target' in oracle:
            reverse_mode = oracle.replace('target_', '')
            if diff_mode == 'diff':
                target = hiddens[:, i, :] - net.reverse_layer(inpt, hiddens[:, i+1, :], reverse_mode, tol_inv) \
                         + net.reverse_layer(inpt, target, reverse_mode, tol_inv)
            elif diff_mode == 'linearized':
                jac_vec = net.jac_reverse_layer(hiddens[:, i+1, :], target-hiddens[:, i+1, :])
                target = hiddens[:, i, :] + jac_vec
            elif diff_mode == 'none':
                target = net.reverse_layer(inpt, target, reverse_mode, tol_inv)
            displacement = target - hiddens[:, i, :]
        elif oracle == 'grad':
            if net.activ == torch.tanh:
                inter = F.linear(inputs[:, i, :], net.rnn.weight_ih_l0, net.rnn.bias_ih_l0) \
                     + F.linear(hiddens[:, i, :], net.rnn.weight_hh_l0, net.rnn.bias_hh_l0)
                grad_activ_inter = 1./torch.cosh(inter)**2
                displacement = grad_activ_inter*displacement
                displacement = F.linear(displacement, net.rnn.weight_hh_l0.t())
        else:
            raise NotImplementedError
        l2_norms.append(torch.norm(displacement).item())

    return l2_norms



