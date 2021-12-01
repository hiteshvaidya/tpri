import sys
import os
import numpy as np
import time
import argparse
from pandas import DataFrame
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


sys.path.append('..')
parser = argparse.ArgumentParser(description='run_exp_on_cluster')
parser.add_argument('--gpu', default=1, type=int)
parser.add_argument('--slice', default=0, type=int)
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

from exp.exp_neck import train_incrementally, check_stop
from pipeline.grid_search import grid_search, set_cfg
from exp.plot_tools import plot, nice_writing

params_plot_heatmap = {'axes.labelsize': 45,
                       'legend.fontsize': 70,
                       'xtick.labelsize': 40,
                       'ytick.labelsize': 40,
                       'lines.linewidth': 2,
                       'text.usetex': True,
                       'figure.figsize': (12.5, 9.5)}
plt.rcParams.update(params_plot_heatmap)



size_chunks = [1, 2, 4, 16, 56, 196]
hidden_sizes = [2**i for i in range(4, 10, 1)]

lr_range = [10**(-i) for i in range(7)]
lr_targ_range = [10**(-i) for i in range(4)]


data_cfgs = [dict(dataset='MNIST', batch_size=512, size_chunk=k) for k in size_chunks]
model_cfgs = [[dict(network='rnn', loss='cross_entropy', input_size=k, hidden_size=j, dim_output=10)
               for j in hidden_sizes] for k in size_chunks]
optim_cfgs = [
             dict(oracle='target_optim', max_iter=400, lr=1e-1, lr_target=lr_range, reg=1e0, diff_mode='linearized'),
             dict(oracle='grad', max_iter=400, lr=lr_range)
             ]
exp_cfgs = [dict(data_cfg=data_cfg, model_cfg=model_cfg, optim_cfg=optim_cfg)
            for i, data_cfg in enumerate(data_cfgs) for model_cfg in model_cfgs[i] for optim_cfg in optim_cfgs]

for exp_cfg in exp_cfgs:
    best_params = grid_search(exp_cfg, train_incrementally, check_stop, 'train_loss')
    exp_cfg = {key: set_cfg(cfg, best_params) for key, cfg in exp_cfg.items()}
    _, _, info_exp = train_incrementally(**exp_cfg)

plot_cvg = False
heatmap = {'Length': list(), 'Width': list(), 'winner': list()}
for i, data_cfg in enumerate(data_cfgs):
    for j, model_cfg in enumerate(model_cfgs[i]):
        info_optim_cvg = DataFrame()
        measure_perf = []
        for optim_cfg in optim_cfgs:
            exp_cfg = dict(data_cfg=data_cfg, model_cfg=model_cfg, optim_cfg=optim_cfg)
            best_params = grid_search(exp_cfg, train_incrementally, check_stop, 'train_loss')
            exp_cfg = {key: set_cfg(cfg, best_params) for key, cfg in exp_cfg.items()}
            _, _, info_exp = train_incrementally(**exp_cfg)

            measure_perf.append(np.trapz(info_exp['accuracy'], info_exp['iteration']))
            if plot_cvg:
                optim_cfg = exp_cfg['optim_cfg']
                name_algo = optim_cfg['oracle']
                if 'diff_mode' in optim_cfg.keys():
                    name_algo = name_algo + '_' + optim_cfg['diff_mode']
                info_exp.update(algo=[name_algo] * len(info_exp['iteration']))
                info_optim_cvg = info_optim_cvg.append(DataFrame(info_exp), ignore_index=True)
        tp_wins = measure_perf[0] > measure_perf[1]

        length = int(784/data_cfg['size_chunk'])
        heatmap['Length'].append(length)
        heatmap['Width'].append(model_cfg['hidden_size'])
        heatmap['winner'].append(tp_wins)

        if plot_cvg:
            infos_to_plot = ['train_loss', 'accuracy']
            fig, axs = plt.subplots(1, len(infos_to_plot), squeeze=False, figsize=(40, 20))
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
                       bbox_to_anchor=(0.5, 1.12)
                       )
            plt.tight_layout()
            plt.show()
            time.sleep(0.5)

cmap = sns.color_palette("colorblind")
colors = (cmap[1], cmap[0])
cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))

heatmap = DataFrame(heatmap)
heatmap = heatmap.pivot('Width', 'Length',  'winner')
print(heatmap)
ax = sns.heatmap(heatmap, linewidth=1.,  cmap=cmap)
zm = np.ma.masked_less(heatmap.values, 0.5)
x= np.arange(len(heatmap.columns)+1)
y= np.arange(len(heatmap.index)+1)
plt.pcolor(x, y, zm, hatch='/', alpha=0., linewidth=0.1)

zm = np.ma.masked_less(~heatmap.values, 0.5)
x= np.arange(len(heatmap.columns)+1)
y= np.arange(len(heatmap.index)+1)
plt.pcolor(x, y, zm, hatch='.', alpha=0., linewidth=0.1)

ax.set_xticks(np.arange(heatmap.shape[1]+1), minor=True)
ax.set_yticks(np.arange(heatmap.shape[1]+1), minor=True)
ax.grid(True, which="minor", color="w", linewidth=2)
ax.tick_params(which="minor", left=False, bottom=False)

fig = ax.get_figure()
plt.gca().invert_yaxis()
fig.canvas.draw()
plt.yticks(rotation='horizontal')
plt.xticks(rotation='horizontal')
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0.25,0.75])
colorbar.set_ticklabels(['BP', 'TP'])

plt.show()

