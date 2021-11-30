import math
import torch
import time
from copy import deepcopy

from src.optim.custom_algos import CustomSGD, CustomAdam
from pipeline.save_load import device


def run_optim(loss, regularization, net, train_data, test_data, nb_iter_per_log,
              oracle='grad', lr=0.1, max_iter=100,
              momentum=0., reg=0., diff_mode='linearized',
              algo='sgd', tol_inv=1e-3,
              lr_reverse=0.1, noise=0.,
              lr_target=0.1,
              input=None, aux_vars=None, log=None,
              verbose=True, logging=True, averaging=0):
    saved_params = []
    aux_net = deepcopy(net) if averaging > 0 else None
    nesterov = True if momentum != 0. else False
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
        log = collect_info(log, loss, regularization, net, aux_vars, train_data, test_data, init_iter, verbose, logging,
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
                                     verbose, logging=logging, averaging=averaging, aux_net=aux_net)
                        print('Fail inverse')
                        log['train_loss'][-1] = float('nan')
                        break

                last_state = deepcopy(output.data)
                last_state.requires_grad = True
                out = loss(net.predict(last_state), label)
                out.backward()
                target = last_state - lr_target * last_state.grad
                net.target_prop(input, states, target, reverse_mode=reverse_mode, diff_mode=diff_mode,
                                tol_inv=tol_inv)
                if reverse_mode == 'auto_enc':
                    optimizer.step(0)
                else:
                    optimizer.step()
                if averaging:
                    pass
            if averaging > 0:
                if len(saved_params) > averaging:
                    saved_params.pop(0)
                saved_params.append([params for params in net.parameters()])
            if iteration % nb_iter_per_log == 0:
                collect_info(log, loss, regularization, net, aux_vars, train_data, test_data, iteration, verbose,
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
                     logging=logging, averaging=averaging, saved_params=saved_params)

    output = net.state_dict()
    if averaging == 0:
        aux_vars = optimizer.state_dict()
    else:
        aux_vars = [optimizer.state_dict(), saved_params]

    return output, aux_vars, log


def collect_info(log, loss, regularization, net, aux_vars, train_data, test_data, iteration, verbose, logging,
                 averaging=0, saved_params=None, aux_net=None):
    log_header = 'Iter\t\t train loss '
    log_format = '{:0.2f} \t\t {:0.6f} '
    if averaging>0:
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
                else:
                    raise NotImplementedError

        info.update(test_loss=test_loss.item())
        info.update(accuracy=100 * correct / total)

    if verbose:
        if iteration == 0:
            print(log_header)
        print(log_format.format(*list(info.values())))

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



