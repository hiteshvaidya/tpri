import os
from matplotlib import pyplot as plt

from exp.plot_tools import plot, nice_writing
from exp.exp_neck import run_exp


def compa_lr_targ():
    max_iter = 400
    data_cfg = dict(dataset='MNIST', vectorize=True, seed=1, nb_train=60000, batch_size=16, normalize=False)
    model_cfg = dict(network='rnn', loss='cross_entropy', reg_param=0., input_size=1, hidden_size=100, dim_output=10)
    optim_cfgs = [dict(oracle='target_optim', max_iter=max_iter,  lr=1e-1, lr_target=lr, reg=1., diff_mode='linearized'
                       ) for lr in [10**i for i in range(-6, -1)]
                  ]
    info_exp = run_exp(data_cfg, model_cfg, optim_cfgs, add_param='lr_target')
    return info_exp


def compa_lr():
    max_iter = 400
    data_cfg = dict(dataset='MNIST', vectorize=True, seed=1, nb_train=60000, batch_size=16, normalize=False)
    model_cfg = dict(network='rnn', loss='cross_entropy', reg_param=0., input_size=1, hidden_size=100, dim_output=10)
    optim_cfgs = [dict(oracle='target_optim', max_iter=max_iter,  lr=lr, lr_target=1e-4, reg=1.,
                       diff_mode='linearized'
                       ) for lr in [10**i for i in range(-4, 1)]
                  ]
    info_exp = run_exp(data_cfg, model_cfg, optim_cfgs, add_param='lr')
    return info_exp


def compa_reg():
    max_iter = 400
    data_cfg = dict(dataset='MNIST', vectorize=True, seed=1, nb_train=60000, batch_size=16, normalize=False)
    model_cfg = dict(network='rnn', loss='cross_entropy', reg_param=0., input_size=1, hidden_size=100, dim_output=10)
    optim_cfgs = [dict(oracle='target_optim', max_iter=max_iter,  lr=1e-1, lr_target=1e-4, reg=reg,
                       diff_mode='linearized'
                       ) for reg in [10**i for i in range(-1, 4)]
                  ]
    info_exp = run_exp(data_cfg, model_cfg, optim_cfgs, add_param='reg')
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
    params = {'axes.labelsize': 55,
              'legend.fontsize': 60,
              'xtick.labelsize': 60,
              'ytick.labelsize': 60,
              'lines.linewidth': 10,
              'text.usetex': True,
              'figure.figsize': (11.5, 9.5)}

    plt.rcParams.update(params)
    exp_to_run = dict(lr_targ_compa=compa_lr_targ, lr_compa=compa_lr, reg_compa=compa_reg)
    info_exp = exp_to_run[exp]()

    fig = plot(info_exp, 'accuracy', with_pal=False)
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()

    for i in range(1, len(labels)):
        labels[i] = nice_writing[labels[i]]
    ax.get_legend().remove()

    if exp == 'lr_targ_compa':
        labels[0] = r'$\gamma_h$'
    elif exp == 'lr_compa':
        labels[0] = r'$\gamma_\theta$'
    elif exp == 'reg_compa':
        labels[0] = r'$r$'
    fig.legend(handles=handles, labels=labels,
               loc='center', bbox_to_anchor=(1.2, 0.5))
    plt.show()

    exp_folder = os.path.dirname(os.path.abspath(__file__))
    plots_folder = os.path.join(exp_folder, 'plots')
    fig.savefig(os.path.join(plots_folder, 'exp_' + exp + '.pdf'), format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    paper_plots('lr_targ_compa')
    paper_plots('lr_compa')
    paper_plots('reg_compa')



#