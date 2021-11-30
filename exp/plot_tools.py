import seaborn as sns
import os
from matplotlib import pyplot as plt

algos = ['target_optim_linearized', 'grad','target_auto_enc_diff',  'target_optim_diff', 'gradadam', 'target_optimadam']

exp_folder = os.path.dirname(os.path.abspath(__file__))
plots_folder = os.path.join(exp_folder, 'plots')

cmap = sns.color_palette("colorblind")
palette = {key:value for key,value in zip(algos, cmap)}
dash_styles = ["",
               (4, 1.5),
               (1, 1),
               (3, 1, 1.5, 1),
               (5, 1, 1, 1),
               (5, 1, 2, 1, 2, 1),
               (2, 2, 3, 1.5),
               (1, 2.5, 3, 1.2),
               (1, 2.5, 3, 1.2)
               ]
dashes = {key:value for key,value in zip(algos, dash_styles[:len(algos)])}

params_plot_heatmap = {'axes.labelsize': 55,
                       'legend.fontsize': 70,
                       'xtick.labelsize': 50,
                       'ytick.labelsize': 50,
                       'lines.linewidth': 2,
                       'text.usetex': True,
                       'figure.figsize': (11.5, 9.5)}

scaled = ['train_loss', 'test_loss']
nice_writing = dict(train_loss='Train Loss', test_loss='Test Loss',
                    inner_stepsize='Param step-size', param_stepsize='Outer step-size',
                    state_stepsize='State step-size', iteration='Iterations',
                    objective='Obj', accuracy='Accuracy',
                    SoftSGD='MorSGD', SGD='SGD',
                    grad='BP', target_optim_diff='DTP', target_auto_enc_diff='DTP-PI',
                    gradadam='BP Adam',
                    lr=r'Stepsize $\gamma_\theta$',
                    target_optim_linearized='TP',
                    reg=r'Regularization $\kappa$',
                    time='Time in s',
                    target_optimadam='DTP-RI-Adam'
                    )
nice_writing.update({'target_optim_1e-06':'1e-06'})
nice_writing.update({'target_optim_1e-05':'1e-05'})
nice_writing.update({'target_optim_1e-04':'1e-04'})
nice_writing.update({'target_optim_1e-03':'1e-03'})
nice_writing.update({'target_optim_1e-02':'1e-02'})


def plot(info_exp, info_to_plot, ax_to_plot=None, with_pal=True):

    x_axis = 'Time in s' if 'time' in info_exp.columns or 'Time in s' in info_exp.columns else 'Iterations'

    info_exp.rename(columns=nice_writing, inplace=True)
    info_to_plot = nice_writing[info_to_plot]
    if ax_to_plot is None:
        fig = plt.figure()
        ax = sns.lineplot(x=x_axis, y=info_to_plot, hue='algo', style='algo', data=info_exp,
                          palette=palette, dashes=dashes
                          )
    else:
        fig = plt.gcf()
        if with_pal:
            ax = sns.lineplot(x=x_axis, y=info_to_plot, hue='algo', style='algo', data=info_exp, ax=ax_to_plot,
                              palette=palette, dashes=dashes
                              )
        else:
            ax = sns.lineplot(x=x_axis, y=info_to_plot, hue='algo', style='algo', data=info_exp, ax=ax_to_plot)
    if info_to_plot == 'Accuracy':
        if max(info_exp[info_to_plot]) < 25:
            ax.set_ylim(0, 25)
            ax.locator_params(axis='y', nbins=4)
        elif max(info_exp[info_to_plot]) < 40:
            ax.set_ylim(0, 40)
            ax.locator_params(axis='y', nbins=4)
        else:
            ax.set_ylim(0, 100)
            ax.locator_params(axis='y', nbins=4)
    else:
        pass
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    return fig



