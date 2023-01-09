from pandas import DataFrame
from matplotlib import pyplot as plt
import argparse
import os
import sys
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm

sys.path.append('..')
from exp.plot_tools import nice_writing, palette, dashes
time = False


def define_exp(exp='grad_thru_layers'):
    if exp == 'grad_thru_layers':
        data_cfg = dict(dataset='MNIST', vectorize=True, seed=1, nb_train=60000, batch_size=16, normalize=False)
        model_cfg = dict(network='rnn', loss='cross_entropy', reg_param=0., input_size=1, hidden_size=100, dim_output=10)
        max_iter = 4000
        optim_cfgs = [
                      dict(oracle='target_optim', max_iter=max_iter, lr=1e-1,
                           lr_target=1e-4, reg=1.,
                           diff_mode='linearized',
                           log_target_norms=True
                           ),
                      dict(oracle='grad', max_iter=max_iter, lr=1e-5, log_target_norms=True),
                      ]
    elif exp == 'grad_spec_rad_along_iters':
        data_cfg = dict(dataset='MNIST', vectorize=True, seed=1, nb_train=60000, batch_size=16, normalize=False)
        model_cfg = dict(network='rnn', loss='cross_entropy', reg_param=0., input_size=1, hidden_size=100, dim_output=10)
        max_iter = 12000
        optim_cfgs = [
                      dict(oracle='target_optim', max_iter=max_iter, lr=1e-1,
                           lr_target=1e-4, reg=1.,
                           diff_mode='linearized',
                           log_grad_norms=True,
                           log_spec_rad=True
                           ),
                      dict(oracle='grad', max_iter=max_iter, lr=1e-5, log_grad_norms=True, log_spec_rad=True),
                      ]
    else:
        raise NotImplementedError
    return data_cfg, model_cfg, optim_cfgs


def grad_behavior_exp(exp):
    data_cfg, model_cfg, optim_cfgs = define_exp(exp)

    params = {'axes.labelsize': 20,
              'legend.fontsize': 20,
              'xtick.labelsize': 15,
              'ytick.labelsize': 15,
              'lines.linewidth': 5,
              'text.usetex': True}
    plt.rcParams.update(params)

    exp_folder = os.path.dirname(os.path.abspath(__file__))
    plots_folder = os.path.join(exp_folder, 'plots')

    info_optim_cvg = DataFrame()
    for i, optim_cfg in enumerate(optim_cfgs):
        _, _, info_exp = train_incrementally(data_cfg, model_cfg, optim_cfg)

        if exp == 'grad_thru_layers':
            fig = plt.figure()
            l2_norms = info_exp['l2_norms']

            if optim_cfg['oracle'] == 'target_optim':
                l2_norms = np.array(l2_norms)
                l2_norms = l2_norms + 1e-12
                ax = sns.heatmap(l2_norms, norm=LogNorm())
            else:
                ax = sns.heatmap(l2_norms)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Iterations')
            xlabels = ax.get_xticklabels()
            xlabels = xlabels[::-1]
            ax.set_xticklabels(xlabels)
            plt.yticks(rotation=0)
            ax.set_yticklabels([400 * i for i in range(11)])

            plt.show()
            if optim_cfg['oracle'] == 'target_optim':
                fig.suptitle('TP', size=24, y=1.05)
                plt.tight_layout()
                fig.savefig(os.path.join(plots_folder, 'tp_l2_norms.pdf'), format='pdf', bbox_inches='tight')
            else:
                fig.suptitle('BP', size=24, y=1.05)
                plt.tight_layout()
                fig.savefig(os.path.join(plots_folder, 'bp_l2_norms.pdf'), format='pdf', bbox_inches='tight')

        algo_name = optim_cfg['oracle']
        if algo_name == 'target_optim':
            algo_name += '_linearized'
        info_exp.update(algo=[algo_name] * len(info_exp['iteration']))
        info_exp = DataFrame(info_exp)

        info_optim_cvg = info_optim_cvg.append(info_exp, ignore_index=True)

    if exp == 'grad_spec_rad_along_iters':
        params = {'legend.fontsize': 55,
                  'axes.labelsize': 60,
                  'xtick.labelsize': 60,
                  'ytick.labelsize': 60}
        params.update({'lines.linewidth': 10, 'text.usetex': True, })
        plt.rcParams.update(params)

        info_optim_cvg.rename(columns=nice_writing, inplace=True)

        fig = plt.figure(figsize=(11.5, 9.5))
        sns.lineplot(x='Iterations', y='Oracle Dir. Norm', data=info_optim_cvg, hue='algo', style='algo',
                     palette=palette, dashes=dashes)
        handles, labels = plt.gca().get_legend_handles_labels()
        for i in range(1, len(labels)):
            labels[i] = nice_writing[labels[i]]
        plt.gca().get_legend().remove()
        plt.yscale('log')
        fig.legend(handles=handles[1:], labels=labels[1:],
                   loc='upper center',
                   ncol=len(labels) - 1,
                   bbox_to_anchor=(0.5, 1.2), handletextpad=0.3, columnspacing=0.5,
                   )
        plt.show()
        fig.savefig(os.path.join(plots_folder, 'grad_norms_iters.pdf'), format='pdf', bbox_inches='tight')

        fig = plt.figure(figsize=(11.5, 9.5))
        sns.lineplot(x='Iterations', y='Spectral Radius', data=info_optim_cvg, hue='algo', style='algo',
                     palette=palette, dashes=dashes)
        plt.gca().get_legend().remove()
        fig.legend(handles=handles[1:], labels=labels[1:],
                   loc='upper center',
                   ncol=len(labels) - 1,
                   bbox_to_anchor=(0.5, 1.2), handletextpad=0.3, columnspacing=0.5,
                   )
        plt.show()

        fig.savefig(os.path.join(plots_folder, 'spec_rad_iters.pdf'), format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run_exp_on_cluster')
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    from exp.exp_neck import train_incrementally

    for exp in ['grad_spec_rad_along_iters', 'grad_spec_rad_along_iters']:
        grad_behavior_exp(exp)







