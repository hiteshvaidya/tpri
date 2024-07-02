import os
import sys
from matplotlib import pyplot as plt
from exp.plot_tools import plot, nice_writing
from exp.exp_neck import run_exp


def add_task(len=30, with_target_auto_enc=False):
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
    if with_target_auto_enc:
        optim_cfgs.append(dict(oracle='target_auto_enc', max_iter=max_iter, lr=1e-2,
                               lr_target=1e-1, lr_reverse=1e-3, noise=0., momentum=0.9, diff_mode='diff'))
    info_exp = run_exp(data_cfg, model_cfg, optim_cfgs)
    info_exp = info_exp[info_exp['iteration'] >= 200]
    return info_exp


def tempord_task(len, with_target_auto_enc=False):
    data_cfg = dict(dataset='temp_ord_1bit', seed=1, max_length=len, fixed_length=True)
    model_cfg = dict(network='rnn', loss='cross_entropy', input_size=6, dim_output=4, hidden_size=100)
    if len==120:
        max_iter = 48000 if not with_target_auto_enc else 40000
        optim_cfgs = [
                      dict(oracle='target_optim', max_iter=max_iter, lr=1e-2,
                           lr_target=1e-2, reg=1., diff_mode='linearized'),
                      dict(oracle='grad', max_iter=max_iter, momentum=0.9, lr=1e-5),
                      # dict(oracle='grad', max_iter=max_iter, lr=1e-4, algo='adam')
                      ]
        if with_target_auto_enc:
            optim_cfgs.append(dict(oracle='target_auto_enc', max_iter=max_iter, lr=1e-2,
                                   lr_target=1e-1, lr_reverse=1e-3, noise=0., momentum=0.9, diff_mode='diff'))
    else:
        max_iter = 40000
        optim_cfgs = [
            dict(oracle='target_optim', max_iter=max_iter, lr=1e-1,
                 lr_target=1e-2, reg=10., diff_mode='linearized'),
            dict(oracle='grad', max_iter=max_iter, momentum=0.9, lr=1e-5),
            # dict(oracle='grad', max_iter=max_iter, lr=1e-4, algo='adam')
        ]
        if with_target_auto_enc:
            optim_cfgs.append(dict(oracle='target_auto_enc', max_iter=max_iter, lr=1e-2,
                                   lr_target=1e-1, lr_reverse=1e-3, noise=0., momentum=0.9, diff_mode='diff'))
    info_exp = run_exp(data_cfg, model_cfg, optim_cfgs)
    return info_exp


def mnist_seq(permut=False, time=False, momentum=None, with_target_auto_enc=False):
    data_cfg = dict(dataset='MNIST', vectorize=True, seed=1, nb_train=60000, batch_size=16, normalize=False)
    if permut:
        data_cfg.update(with_permut=True)
    model_cfg = dict(network='rnn', loss='cross_entropy', reg_param=0., input_size=1, hidden_size=100, dim_output=10)
    max_iter = 12000 if time else 40000
    lr_grad = 1e-4 if permut else 1e-5
    lr_grad = 1e-6 if time else lr_grad
    max_iter_grad = 13*max_iter if time else max_iter
    optim_cfgs = [dict(oracle='target_optim', max_iter=max_iter, lr=1e-1,
                       lr_target=1e-4, reg=1.,
                       diff_mode='linearized'
                       ),
                  dict(oracle='grad', max_iter=max_iter_grad, lr=lr_grad),
                  ]
    if with_target_auto_enc:
        optim_cfgs.append(dict(oracle='target_auto_enc', max_iter=max_iter, lr=1e-2,
                               lr_target=1e-7, lr_reverse=1e-8, noise=0.1, diff_mode='diff'))
    if momentum is not None:
        for optim_cfg in optim_cfgs:
            optim_cfg.update(momentum=momentum)
    info_exp = run_exp(data_cfg, model_cfg, optim_cfgs, time)
    return info_exp


def compa_cifar():
    data_cfg = dict(dataset='CIFAR', batch_size=16)
    model_cfg = dict(network='rnn', loss='cross_entropy', input_size=1, hidden_size=100, dim_output=10)
    max_iter = 8000
    optim_cfgs = [dict(oracle='target_optim', max_iter=max_iter, lr=1e-2, lr_target=1e-2, reg=1e1, diff_mode='linearized'),
                  dict(oracle='grad', max_iter=max_iter, lr=1e-3)
                  ]
    info_exp = run_exp(data_cfg, model_cfg, optim_cfgs)
    return info_exp


def penn_tree():
    data_cfg = dict(dataset='penn_treebank', nb_train=40000, batch_size=512, max_length=64)
    model_cfg = dict(network='rnn_text', loss='cross_entropy_seq', embedding_dim=1024, input_size=1, hidden_size=256,
                     dim_output=10000, freeze_embedding=True)
    max_iter = 24000
    optim_cfgs = [
                  dict(oracle='grad', max_iter=max_iter,  lr=1e-2),
                  dict(oracle='target_optim', max_iter=max_iter, diff_mode='linearized',
                       lr=1e-1, lr_target=1e-3, reg=1e0)
                  ]
    info_exp = run_exp(data_cfg, model_cfg, optim_cfgs)
    return info_exp


def compa_gru():
    data_cfg = dict(dataset='FashionMNIST', batch_size=16)
    model_cfg = dict(network='gru', loss='cross_entropy', input_size=1, hidden_size=100, dim_output=10)
    max_iter = 8000
    optim_cfgs = [
        dict(oracle='target_optim', max_iter=max_iter, lr=1e-1, lr_target=1e-2, reg=1e0),
        dict(oracle='grad', max_iter=max_iter, lr=1e-2)
        ]
    info_exp = run_exp(data_cfg, model_cfg, optim_cfgs)
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
    if exp == 'synth':
        info_exps = [tempord_task(60), tempord_task(120), add_task(30)]
    elif exp == 'images_and_text':
        info_exps = [mnist_seq(), mnist_seq(permut=True), compa_cifar(),
                     compa_gru(), penn_tree()]
    elif exp == 'compa_auto_enc':
        info_exps = [
            tempord_task(60, with_target_auto_enc=True),
            tempord_task(120, with_target_auto_enc=True),
            add_task(30, with_target_auto_enc=True),
            mnist_seq(with_target_auto_enc=True),
            mnist_seq(permut=True, with_target_auto_enc=True)
        ]
    elif exp == 'mnist_time':
        info_exps = [mnist_seq(time=True)]
    elif exp == 'mnist_mom':
        info_exps = [mnist_seq(momentum=0.9)]
    elif exp == 'compa_inv_grad_diff_targ':
        info_exps = [compa_inv_grad_diff_targ()]
    else:
        raise ValueError

    if exp in ['compa_inv_grad_diff_targ', 'mnist_mom', 'mnist_time']:
        infos_to_plot = ['accuracy']
    else:
        infos_to_plot = ['train_loss', 'accuracy']

    if exp in ['compa_inv_grad_diff_targ', 'mnist_mom', 'mnist_time']:
        params = {'legend.fontsize': 55,
                  'axes.labelsize': 60,
                  'xtick.labelsize': 60,
                  'ytick.labelsize': 60}
    else:
        params = {'axes.labelsize': 65,
                  'legend.fontsize': 70,
                  'xtick.labelsize': 60,
                  'ytick.labelsize': 60}

    params.update({'lines.linewidth': 10, 'text.usetex': True,})

    plt.rcParams.update(params)

    if exp in ['compa_inv_grad_diff_targ', 'mnist_mom', 'mnist_time']:
        figsize = (11.5, 9.5)
    elif exp == 'synth':
        figsize = (40, 20)
    elif exp == 'images_and_text':
        figsize = (55, 20)
    elif exp == 'compa_auto_enc':
        figsize = (55, 20)
    else:
        raise NotImplementedError
    fig, axs = plt.subplots(len(infos_to_plot), len(info_exps), squeeze=False, figsize=(figsize))

    for i, info_to_plot in enumerate(infos_to_plot):
        for j, info_exp in enumerate(info_exps):
            fig = plot(info_exp, info_to_plot, axs[i, j], with_pal=True)
    handles, labels = axs[-1, -1].get_legend_handles_labels()

    for i in range(1, len(labels)):
        labels[i] = nice_writing[labels[i]]

    for i in range(len(infos_to_plot)):
        for j in range(len(info_exps)):
            axs[i, j].get_legend().remove()
            if j > 0:
                axs[i, j].set(ylabel=None)

    top_margin = 1.12 if exp not in ['compa_inv_grad_diff_targ', 'mnist_mom', 'mnist_time'] else 1.2

    fig.legend(handles=handles[1:], labels=labels[1:],
               loc='upper center',
               ncol=len(labels) - 1,
               bbox_to_anchor=(0.5, top_margin), handletextpad=0.3, columnspacing=0.5,
               )
    plt.show()


if __name__ == '__main__':
    paper_plots('synth')
    paper_plots('images_and_text')
    paper_plots('compa_inv_grad_diff_targ')
    paper_plots('compa_auto_enc')
    paper_plots('mnist_time')
