from pandas import DataFrame
from matplotlib import pyplot as plt
import argparse
import os
import sys

sys.path.append('..')
from exp.plot_tools import plot, nice_writing, plots_folder
from exp.exp_neck import check_stop
time = False


def define_exp(task='mnist', length=120):
    if task == 'addtask':
        data_cfg = dict(dataset='addtask', seed=1,  max_length=30, fixed_length=True)
        model_cfg = dict(network='rnn', loss='mean_squared', input_size=2, dim_output=1, hidden_size=100)
        max_iter = 40000

    elif task == 'temp_ord_1bit':
        data_cfg = dict(dataset='temp_ord_1bit', seed=1, max_length=length, fixed_length=True)
        model_cfg = dict(network='rnn', loss='cross_entropy', input_size=6, dim_output=4, hidden_size=100)
        max_iter = 40000

    elif task == 'cifar':
        data_cfg = dict(
                        dataset='CIFAR',
                        batch_size=16,
                        )
        model_cfg = dict(
                        network='rnn',
                        loss='cross_entropy',
                        input_size=1,
                        hidden_size=100,
                        dim_output=10,
                        )
        max_iter = 8000
    elif task == 'mnist':
        data_cfg = dict(
                        dataset='MNIST',
                        batch_size=512,
                        size_chunk=1
                        )
        model_cfg = dict(
                        network='rnn',
                        loss='cross_entropy',
                        input_size=1,
                        hidden_size=128,
                        dim_output=10,
                        )
        max_iter = 400
    else:
        raise NotImplementedError
    optim_cfgs = [
                    dict(oracle='grad',
                         max_iter=max_iter,
                         lr=1e-3,
                         ),
                    dict(
                        oracle='target_optim', max_iter=max_iter,
                        lr=1e-1,
                        lr_target=1e0,
                        reg=1e0,
                        diff_mode='linearized',
                    ),
    ]
    return data_cfg, model_cfg, optim_cfgs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run_exp_on_cluster')
    parser.add_argument('--gpu', default=1, type=int)
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    from exp.exp_neck import train_incrementally

    data_cfg, model_cfg, optim_cfgs = define_exp()

    info_optim_cvg = DataFrame()
    for i, optim_cfg in enumerate(optim_cfgs):
        _, _, info_exp = train_incrementally(data_cfg, model_cfg, optim_cfg)
        name_algo = optim_cfg['oracle']
        if name_algo == 'target_optim' and data_cfg['dataset'] == 'CIFAR':
            name_algo = name_algo + '_linearized'
        if 'algo' in optim_cfg.keys():
            name_algo = name_algo + optim_cfg['algo']
        if 'diff_mode' in optim_cfg.keys():
            name_algo = name_algo + '_' + optim_cfg['diff_mode']
        info_exp.update(algo=[name_algo] * len(info_exp['iteration']))
        print(check_stop(info_exp))

        info_optim_cvg = info_optim_cvg.append(DataFrame(info_exp), ignore_index=True)

    infos_to_plot = ['train_loss', 'accuracy']
    params = {'axes.labelsize': 65,
              'legend.fontsize': 70,
              'xtick.labelsize': 60,
              'ytick.labelsize': 60,
              'lines.linewidth': 10,
              'text.usetex': True,
              'figure.figsize': (10, 5)}
    plt.rcParams.update(params)
    fig, axs = plt.subplots(1, len(infos_to_plot), squeeze=False, figsize=(40, 20))
    for i, info_to_plot in enumerate(infos_to_plot):
        fig = plot(info_optim_cvg, info_to_plot, axs[0, i])
    handles, labels = axs[-1, -1].get_legend_handles_labels()
    for i in range(1, len(labels)):
        labels[i] = nice_writing[labels[i]]
    for i in range(len(infos_to_plot)):
        axs[0, i].get_legend().remove()
    if data_cfg['dataset'] == 'CIFAR':
        axs[0, 1].set_ylim(0, 25)
        axs[0, 1].locator_params(axis='y', nbins=5)
    fig.legend(handles=handles[1:], labels=labels[1:],
                loc='upper center',
                      ncol = len(labels) - 1,
                             bbox_to_anchor = (0.5, 1.12)
    )
    plt.tight_layout()
    plt.show()