import os
import argparse
import sys
from matplotlib import pyplot as plt

sys.path.append('..')

parser = argparse.ArgumentParser(description='run_exp_on_cluster')
parser.add_argument('--exp', default='mnist', type=str)
args = parser.parse_args()


from exp.plot_tools import plot, nice_writing
from exp.exp_neck import run_exp


def add_task(len=30):
    data_cfg = dict(dataset='addtask', seed=1,
                    max_length=len,
                    fixed_length=True)
    model_cfg = dict(network='rnn', loss='mean_squared', input_size=2, dim_output=1, hidden_size=100)
    max_iter = 24000
    optim_cfgs = [
                  dict(oracle='target_optim', max_iter=max_iter, lr=1e-1,
                       lr_target=1e-1, reg=1., diff_mode='linearized'),
                  dict(oracle='grad', max_iter=max_iter, momentum=0.9, lr=1e-3),
                 ]
    info_exp = run_exp(data_cfg, model_cfg, optim_cfgs)
    info_exp = info_exp[info_exp['iteration'] >= 200]
    return info_exp


def tempord_task(len):
    data_cfg = dict(dataset='temp_ord_1bit', seed=1, max_length=len, fixed_length=True)
    model_cfg = dict(network='rnn', loss='cross_entropy', input_size=6, dim_output=4, hidden_size=100)
    if len == 120:
        max_iter = 48000
        optim_cfgs = [
            dict(oracle='target_optim', max_iter=max_iter, lr=1e-2,
                 lr_target=1e-2, reg=1., diff_mode='linearized'),
            dict(oracle='grad', max_iter=max_iter, momentum=0.9, lr=1e-5),
        ]
    else:
        max_iter = 40000
        optim_cfgs = [
            dict(oracle='target_optim', max_iter=max_iter, lr=1e-1,
                 lr_target=1e-2, reg=10., diff_mode='linearized'),
            dict(oracle='grad', max_iter=max_iter, momentum=0.9, lr=1e-5),
        ]
    info_exp = run_exp(data_cfg, model_cfg, optim_cfgs)
    return info_exp


def mnist_seq(permut=False, time=False, momentum=None):
    data_cfg = dict(dataset='MNIST', vectorize=True, seed=1, nb_train=60000, batch_size=16, normalize=False)
    if permut:
        data_cfg.update(with_permut=True)
    model_cfg = dict(network='rnn', loss='cross_entropy', reg_param=0., input_size=1, hidden_size=100, dim_output=10)
    max_iter = 12000 if time else 40000
    lr_grad = 1e-4 if permut else 1e-5
    if time:
        lr_grad = 1e-6
    max_iter_grad = 13*max_iter if time else max_iter
    optim_cfgs = [dict(oracle='target_optim', max_iter=max_iter, lr=1e-1,
                       lr_target=1e-4, reg=1.,
                       diff_mode='linearized'
                       ),
                  dict(oracle='grad', max_iter=max_iter_grad, lr=lr_grad),
                  ]
    if momentum is not None:
        for optim_cfg in optim_cfgs:
            optim_cfg.update(momentum=momentum)
        optim_cfgs[0].update(diff_mode='linearized')
    info_exp = run_exp(data_cfg, model_cfg, optim_cfgs, time)
    return info_exp


def compa_cifar():
    data_cfg = dict(dataset='CIFAR', batch_size=16)
    model_cfg = dict(network='rnn', loss='cross_entropy', input_size=1, hidden_size=100, dim_output=10)
    max_iter = 8000
    optim_cfgs = [
        dict(oracle='target_optim', max_iter=max_iter, lr=1e-2, lr_target=1e-2, reg=1e1, diff_mode='linearized'),
        dict(oracle='grad', max_iter=max_iter, lr=1e-3)
        ]
    info_exp = run_exp(data_cfg, model_cfg, optim_cfgs)
    return info_exp


def compa_gru():
    data_cfg = dict(dataset='FashionMNIST', batch_size=16)
    model_cfg = dict(network='gru', loss='cross_entropy', input_size=1, hidden_size=100, dim_output=10)
    max_iter = 8000
    optim_cfgs = [
        dict(oracle='target_optim', max_iter=max_iter, lr=1e-1, lr_target=1e-2, reg=1e0, diff_mode='lienarized'),
        dict(oracle='grad', max_iter=max_iter, lr=1e-2)
        ]
    info_exp = run_exp(data_cfg, model_cfg, optim_cfgs)

    return info_exp


def compa_lr_targ():
    max_iter = 4000
    data_cfg = dict(dataset='MNIST', vectorize=True, seed=1, nb_train=60000, batch_size=16, normalize=False)
    model_cfg = dict(network='rnn', loss='cross_entropy', reg_param=0., input_size=1, hidden_size=100, dim_output=10)
    optim_cfgs = [dict(oracle='target_optim', max_iter=max_iter,  lr=1e-1, lr_target=lr, reg=1., diff_mode='linearized'
                       ) for lr in [10**i for i in range(-6, -1)]
                  ]
    info_exp = run_exp(data_cfg, model_cfg, optim_cfgs, add_lr=True)
    return info_exp


def compa_inv_grad_diff_targ():
    data_cfg = dict(dataset='MNIST', vectorize=True, seed=1, nb_train=60000, batch_size=16, normalize=False)
    model_cfg = dict(network='rnn', loss='cross_entropy', reg_param=0., input_size=1, hidden_size=100, dim_output=10)
    max_iter = 40000
    optim_cfgs = [
                  dict(oracle='target_optim', max_iter=max_iter, lr=1e-1,
                       lr_target=1e-4, reg=1., diff_mode='linearized'),
                  dict(oracle='target_optim', max_iter=max_iter, lr=1e-1,
                       lr_target=1e-4, reg=1., diff_mode='diff'),
                  dict(oracle='target_auto_enc', max_iter=max_iter, lr=1e-2,
                       lr_target=1e-7, lr_reverse=1e-8, noise=0.1, diff_mode='diff')
                  ]
    info_exp = run_exp(data_cfg, model_cfg, optim_cfgs)
    return info_exp


def paper_plots(exp='synth'):
    infos_to_plot = ['train_loss', 'accuracy'] if exp not in ['compa_inv_grad_diff_targ', 'mnist_mom', 'lr_targ_compa', 'mnist_time']\
        else ['accuracy']

    params = {'axes.labelsize': 65,
              'legend.fontsize': 70,
              'xtick.labelsize': 60,
              'ytick.labelsize': 60,
              'lines.linewidth': 10,
              'text.usetex': True,
              'figure.figsize': (10, 5)}
    if exp in ['compa_inv_grad_diff_targ', 'mnist_mom', 'lr_targ_compa', 'mnist_time']:
        params.update({'legend.fontsize': 55})
        params.update({'axes.labelsize': 55})
        params.update({'xtick.labelsize': 50})
        params.update({'ytick.labelsize': 50})

    plt.rcParams.update(params)

    if exp == 'synth':
        info_exps = [
            tempord_task(60),
            tempord_task(120),
            add_task(30)
        ]
    elif exp == 'images':
        info_exps = [
                     mnist_seq(),
                     mnist_seq(permut=True),
                     compa_cifar(),
                     compa_gru()
                     ]
    elif exp == 'mnist_time':
        info_exps = [mnist_seq(time=True)]
    elif exp == 'mnist_mom':
        info_exps = [mnist_seq(momentum=0.9)]
    elif exp == 'lr_targ_compa':
        info_exps = [compa_lr_targ()]
    elif exp == 'compa_inv_grad_diff_targ':
        info_exps = [compa_inv_grad_diff_targ()]
    else:
        raise ValueError
    if exp in ['compa_inv_grad_diff_targ', 'mnist_mom', 'lr_targ_compa', 'mnist_time']:
        figsize = (11.5, 9.5)
    elif exp == 'synth':
        figsize = (40, 20)
    elif exp == 'images':
        figsize = (50, 20)
    else:
        raise NotImplementedError
    # figsize = (40, 20) if exp not in ['compa_inv_grad_diff_targ', 'mnist_mom', 'lr_targ_compa'] else (11.5, 9.5)
    fig, axs = plt.subplots(len(infos_to_plot), len(info_exps), squeeze=False, figsize=(figsize))

    for i, info_to_plot in enumerate(infos_to_plot):
        for j, info_exp in enumerate(info_exps):
            fig = plot(info_exp, info_to_plot, axs[i, j])
    handles, labels = axs[-1, -1].get_legend_handles_labels()

    for i in range(1, len(labels)):
        labels[i] = nice_writing[labels[i]]

    for i in range(len(infos_to_plot)):
        for j in range(len(info_exps)):
            axs[i, j].get_legend().remove()
    for i in range(len(infos_to_plot)):
        for j in range(1, len(info_exps)):
            axs[i, j].set(ylabel=None)

    top_margin = 1.12 if exp not in ['compa_inv_grad_diff_targ', 'mnist_mom', 'lr_targ_compa', 'mnist_time'] else 1.2

    if exp != 'lr_targ_compa':
        fig.legend(handles=handles[1:], labels=labels[1:],
                   loc='upper center',
                   ncol=len(labels) - 1,
                   bbox_to_anchor=(0.5, top_margin), handletextpad=0.3, columnspacing=0.5,
                   # mode='expand'
                   )
    else:
        labels[0] = r'$\gamma_h$'
        fig.legend(handles=handles, labels=labels,
                   loc='center',
                   bbox_to_anchor=(top_margin, 0.5)
                   )
    plt.show()

paper_plots('synth')
paper_plots('images')
paper_plots('mnist_time')
paper_plots('compa_inv_grad_diff_targ')
paper_plots('mnist_mom')
paper_plots('lr_targ_compa')
