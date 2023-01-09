from pandas import DataFrame
from matplotlib import pyplot as plt
import argparse
import os
import sys

sys.path.append('..')

def define_exp():
    data_cfg = dict(
                    dataset='penn_treebank',
                    nb_train=40000,
                    batch_size=512,
                    max_length=64,
                    )
    model_cfg = dict(
                    network='rnn_text',
                    loss='cross_entropy_seq',
                    embedding_dim=1024,
                    input_size=1,
                    hidden_size=256,
                    dim_output=10000,
                    freeze_embedding=True,
                    )
    max_iter = 24000
    optim_cfgs = [
                    dict(oracle='grad',
                         max_iter=max_iter,
                         lr=1e-2,
                         ),
                    dict(
                        oracle='target_optim', max_iter=max_iter,
                        diff_mode='linearized',
                        lr=1e-1,
                        lr_target=1e-3,
                        reg=1e0,
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
        if name_algo == 'target_optim':
            if 'diff_mode' in optim_cfg.keys() and optim_cfg['diff_mode'] != 'diff':
                name_algo = name_algo
            else:
                name_algo = name_algo + '_linearized'

        if 'algo' in optim_cfg.keys():
            name_algo = name_algo + optim_cfg['algo']
        info_exp.update(algo=[name_algo] * len(info_exp['iteration']))

        info_optim_cvg = info_optim_cvg.append(DataFrame(info_exp), ignore_index=True)

    plot_results = True
    if plot_results:
        from exp.plot_tools import plot, nice_writing

        print(info_optim_cvg[info_optim_cvg['iteration']%1000==0])
        infos_to_plot = ['train_loss', 'accuracy']
        params = {'axes.labelsize': 65,
                  'legend.fontsize': 70,
                  'xtick.labelsize': 60,
                  'ytick.labelsize': 60,
                  'lines.linewidth': 10,
                  'text.usetex': True,
                  'figure.figsize': (10, 5)}
        plt.rcParams.update(params)
        fig, axs = plt.subplots(1, len(infos_to_plot), squeeze=False, figsize=(30, 15))
        for i, info_to_plot in enumerate(infos_to_plot):
            fig = plot(info_optim_cvg, info_to_plot, axs[0, i])
        handles, labels = axs[-1, -1].get_legend_handles_labels()
        for i in range(1, len(labels)):
            labels[i] = nice_writing[labels[i]]
        for i in range(len(infos_to_plot)):
            axs[0, i].get_legend().remove()
        fig.legend(handles=handles[1:], labels=labels[1:],
                   loc='upper center',
                   ncol=len(labels) - 1,
        )
        plt.tight_layout()
        plt.show()
