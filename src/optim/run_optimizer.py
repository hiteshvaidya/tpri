import math
import torch
import time
from copy import deepcopy

from src.optim.custom_algos import CustomSGD
from pipeline.save_load import device


def run_optim(loss, regularization, net, train_data, test_data, nb_iter_per_log,
              oracle='grad', lr=0.1, max_iter=100,
              momentum=0., reg=0., diff_mode='linearized',
              algo='sgd', tol_inv=1e-3,
              lr_reverse=0.1, noise=0.,
              lr_target=0.1,
              input=None, aux_vars=None, log=None,
              verbose=True, logging=True, averaging=0):
    """
    Run an optimization algorithm on the specified problem

    Args:
        loss: (torch.nn.Module) Loss used for the task at hand such as the cross entropy loss
        regularization: (func) regularization on the parameters of the network
        net: (torch.nn.Module) recurrent network
        train_data, test_data: (torch.utils.data.Dataloader) Loaders of training and testing data samples
        oracle: (str) what kind of oracle (either 'grad' for SGD or 'target_prop' for Target Propagation)
        lr: (float) stepsize when updating the weights of the network
                using either a gradient oracle or the direction computed by target propagation
        max_iter: (int) maximal number of iterations
        momentum: (float) momentum to used, see torch.optim.SGD
        reg: (float) regularization to apply when computing regularized inverses
        diff_mode: (str) how targets are propagated in target propagation see src.model.rnn for more details
        algo: (str) which optimization algorithm to use ('sgd' for stochastic updates, 'adam' for Adam updates)
        tol_inv: (float) tolerance parameter when inverting the nonlinear activation functions
        lr_reverse: (float) stepsize to update a reverse layer when the latter is parameterized
        noise: (float) noise to add on the outputs when updating a parameterized reverse layer with a mean squared loss,
                see src.model.rnn for more details
        lr_target: (float) stepsize to compute the first target by a gradient step on the prediction

        input: (net.state_dict) previously computed weights of the network when the experiment
                is reloaded to be run a longer time
        aux_vars: (dict) additional parameters necessary for the optimization
                that were saved in order to run the algorithm for more time
        log: (dict) previous log of the experiment run so far when the experiment is reloaded
        verbose: (bool) whether to print the evolution of the optimization (training loss, test loss, etc...)
        logging: (bool) whether to log the information of the optimization during training
        averaging (int) whether to compute the training/testing loss on an average of some numbers of iterates

    Returns:
        output: (net.parameters) parameters of the optimized network
        aux_vars: (dict) additional parameters used in the optimization process
                that can be saved to restart the optimization from the last iteration
        log: (dict) dictionary containing several measures of performance of the optimization
                algorithm along the iterations

    """
    saved_params = []
    # Used another copy of the newtork when one wants to look at
    # the training/testng loss on an average of the parameters
    aux_net = deepcopy(net) if averaging > 0 else None
    nesterov = True if momentum != 0. else False
    if oracle == 'target_auto_enc':
        # Build a SGD optimizer that updates the network and the reverse layer with two different set of hyperparameters
        forward_params = [param for name, param in net.named_parameters() if 'reverse' not in name]
        backward_params = [param for name, param in net.named_parameters() if 'reverse' in name]
        optimizer = CustomSGD([{'params': forward_params},
                               {'params': backward_params, 'lr': lr_reverse}],
                              lr=lr, momentum=momentum, nesterov=nesterov)
    else:
        if algo == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
        elif algo == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        elif algo == 'adagrad':
            optimizer = torch.optim.Adagrad(net.parameters(), lr=lr)
        else:
            raise NotImplementedError

    # Load previously computed input, aux_vars and log or initialize those parameters if needed
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
    # Main lop below
    while iteration <= max_iter:
        optimizer.zero_grad()
        for input, label in train_data:
            if iteration > max_iter:
                break
            optimizer.zero_grad()
            input = input.to(device)
            label = label.to(device)
            if oracle == 'grad':
                # Classical SGD
                out = loss(net(input), label) + regularization(net)
                out.backward()
                optimizer.step()
            elif 'target' in oracle:
                # Target prop
                # Extract the type of inverse approximation used
                reverse_mode = oracle.replace('target_', '')
                # Compute the sequence of hidden states
                states, output = net.rep(input)
                if reverse_mode == 'auto_enc':
                    # Update the parameterized inverse layer if a parameterized inverse layer was used
                    net.update_reverse(input, states, sigma_noise=noise, reverse_mode=reverse_mode)
                    optimizer.step(1)
                elif reverse_mode in ['optim', 'optim_variant']:
                    # Compute the regularized inverse
                    net.update_reverse(reg=reg, reverse_mode=reverse_mode)
                    # If the inversion failed, save current results and exit
                    if hasattr(net, 'inv_layer') and net.inv_layer is None:
                        collect_info(log, loss, regularization, net, aux_vars, train_data, test_data, iteration,
                                     verbose, logging=logging, averaging=averaging, aux_net=aux_net)
                        print('Fail inverse')
                        log['train_loss'][-1] = float('nan')
                        break
                # Make one step on the last hidden state to define the initial target
                last_state = deepcopy(output.data)
                last_state.requires_grad = True
                out = loss(net.predict(last_state), label)
                out.backward()
                target = last_state - lr_target * last_state.grad
                # Propagate the targets and compute the associated weight update directions
                net.target_prop(input, states, target, reverse_mode=reverse_mode, diff_mode=diff_mode,
                                tol_inv=tol_inv)
                # Make one step along the directions computed
                if reverse_mode == 'auto_enc':
                    optimizer.step(0)
                else:
                    optimizer.step()
                if averaging:
                    pass

            # Save previous parameters if an average of the parameters is used
            if averaging > 0:
                if len(saved_params) > averaging:
                    saved_params.pop(0)
                saved_params.append([params for params in net.parameters()])

            # Collect the information on the optimization process
            if iteration % nb_iter_per_log == 0:
                collect_info(log, loss, regularization, net, aux_vars, train_data, test_data, iteration, verbose,
                             logging=logging, averaging=averaging, saved_params=saved_params, aux_net=aux_net)
                if assess_time:
                    print('Time for {0} iter: {1}s'.format(nb_iter_per_log, time.time() - start_time))
                    assess_time = False
            iteration += 1

        # Check whether the algorithm diverged
        stopped = check_stop(log)
        if stopped is not None:
            print('{0} {1}'.format(optimizer.__class__.__name__, stopped))
            break

    # Save final computations
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
    """
    Collect the information on the objective such as training/testing losses
    """
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
    """
    Check from the log if the algorithm diverged
    """
    if log['train_loss'][-1] > 2 * log['train_loss'][0] or math.isnan(log['train_loss'][-1]):
        print(log['train_loss'][-1])
        stopped = 'has diverged'
    else:
        stopped = None
    return stopped



