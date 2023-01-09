import warnings
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
from torch import autograd


class InnerGD:
    def __init__(self, max_iter, line_search=True, max_line_search=5):
        self.max_iter = max_iter

        self.line_search = line_search
        self.max_line_search = max_line_search
        self.goldstein_slope_factor = 0.5
        self.goldstein_step_factor = 0.5

    def solve(self, func, var, step_size):
        is_layer = isinstance(var, torch.nn.Module)

        for k in range(self.max_iter):
            if k == 0 and self.line_search:
                var, step_size = goldstein_line_search(func, var, self.max_line_search,
                                                       self.goldstein_step_factor, self.goldstein_slope_factor,
                                                       step_size)
            else:
                func_var = func(var)
                if is_layer:
                    grads = autograd.grad(func_var, var.parameters())
                else:
                    grads = autograd.grad(func_var, var)
                descent_dirs = [-grad / torch.norm(grad) for grad in grads]
                if is_layer:
                    for param, descent_dir in zip(var.parameters(), descent_dirs):
                        param.data = param + step_size * descent_dir
                else:
                    var = var + step_size * descent_dirs[0]
        return var, step_size


class InnerBarzilaiBorwein:
    def __init__(self, max_iter, max_line_search=5, step_size_choice=1, init_stepsize=1):
        self.max_iter = max_iter
        self.max_line_search = max_line_search
        self.goldstein_slope_factor = 0.5
        self.goldstein_step_factor = 0.5
        self.step_size_choice = step_size_choice
        self.init_step_size = init_stepsize

    def solve(self, func, var, log=False):
        is_layer = isinstance(var, torch.nn.Module)

        var_prev = deepcopy(var)

        func_val = func(var)
        grads = autograd.grad(func_val, var.parameters()) if is_layer else autograd.grad(func_val, var)
        grads_prev = deepcopy(grads)
        # if log:
        #     valfuncs = [func_val]
        #     gradnorms = [sum([torch.norm(grad) for grad in grads])]
        for k in range(self.max_iter):
            if k == 0:
                var, step_size = goldstein_line_search(func, var, self.max_line_search,
                                                       self.goldstein_step_factor, self.goldstein_slope_factor,
                                                       init_step_size=self.init_step_size)
            else:
                func_val = func(var)
                grads = autograd.grad(func_val, var.parameters()) if is_layer else autograd.grad(func_val, var)

                grad_diffs = [grad - grad_prev for grad, grad_prev in zip(grads, grads_prev)]
                if is_layer:
                    var_diffs = [param - param_prev for param, param_prev in zip(var.parameters(), var_prev.parameters())]
                else:
                    var_diffs = [var - var_prev]

                if self.step_size_choice == 1:
                    step_size = sum([torch.sum(var_diff*grad_diff)
                                     for var_diff, grad_diff in zip(var_diffs, grad_diffs)])
                    step_size = step_size/sum([torch.sum(grad_diff*grad_diff) + 1e-12 for grad_diff in grad_diffs])
                elif self.step_size_choice == 2:
                    step_size = sum([torch.sum(var_diff * var_diff) for var_diff in var_diffs])
                    step_size = step_size/sum([torch.sum(var_diff * grad_diff)
                                              for var_diff, grad_diff in zip(var_diffs, grad_diffs)])
                else:
                    raise NotImplementedError
                var_prev = deepcopy(var) if is_layer else deepcopy(var.data)
                grads_prev = deepcopy(grads)

                if is_layer:
                    for param, grad in zip(var.parameters(), grads):
                        param.data = param - step_size * grad
                else:
                    var = var - step_size * grads[0]
            if step_size <1e-6:
                break

        return var




def goldstein_line_search(func, var, max_line_search, step_factor, slope_factor, init_step_size=None):
    step_size = 10 if init_step_size is None else init_step_size
    is_layer = isinstance(var, torch.nn.Module)
    value = func(var)
    grads = autograd.grad(value, var.parameters()) if is_layer else autograd.grad(value, var)
    descent_dirs = [-grad /(torch.norm(grad) + 1e-12) for grad in grads]

    fixed_var = deepcopy(var)
    step_size = 2 * step_size
    local_slope = sum([torch.sum(grad*descent_dir) for grad, descent_dir in zip(grads, descent_dirs)])
    decrease_target = -slope_factor * local_slope
    for i in range(max_line_search):
        if is_layer:
            for param, fix_param, descent_dir in zip(var.parameters(), fixed_var.parameters(), descent_dirs):
                param.data = fix_param + step_size * descent_dir
        else:
            var = fixed_var + step_size * descent_dirs[0]
        next_value = func(var)
        if next_value <= value - step_size * decrease_target:
            break
        else:
            step_size = step_factor * step_size
        if step_size < 1e-12:
            type_var = 'param' if is_layer else 'state'
            warnings.warn('Line search failed for ' + type_var)
    return var, step_size


