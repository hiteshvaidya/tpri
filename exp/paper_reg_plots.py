import argparse
import os
import sys
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import DataFrame

parser = argparse.ArgumentParser(description='run_exp_on_cluster')
parser.add_argument('--gpu', default=0, type=int)
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
sys.path.append('..')

from pipeline.heatmap import compute_heatmap
from exp.exp_neck import check_stop, train_incrementally
from exp.plot_tools import params_plot_heatmap

max_iter = 400
lr_range = [10**i for i in range(-4, 2)]
reg_range = [10**i for i in range(-4, 4)]

data_cfg = dict(dataset='MNIST', vectorize=True, seed=1, nb_train=60000, batch_size=16, normalize=False)
model_cfg = dict(network='rnn', loss='cross_entropy', reg_param=0., input_size=1, hidden_size=100, dim_output=10)
optim_cfg = dict(oracle='target_optim', max_iter=max_iter, lr=lr_range, lr_target=1e-4, reg=reg_range, diff_mode='linearized')
exp_cfg = dict(data_cfg=data_cfg, model_cfg=model_cfg, optim_cfg=optim_cfg)

plt.rcParams.update(params_plot_heatmap)

heatmap = compute_heatmap(exp_cfg, 'lr', 'reg', train_incrementally, check_stop, 'train_loss')

aux = DataFrame(heatmap)
aux = aux.fillna(10**(-9))
heatmap = aux
fig = plt.figure()
n_colors = len(heatmap['measure'].unique())
pal = sns.color_palette("rocket", n_colors=n_colors)
ax = sns.scatterplot('reg', 'lr', data=heatmap.to_dict(), hue='measure', size='measure',
                     sizes=(50, 1000), palette=pal)
ax.get_legend().remove()
ax.set(xscale="log")
ax.set(yscale="log")
ax.set_ylabel(r'Stepsize $\gamma_\theta$')
ax.set_xlabel(r'Regularization $r$')
plt.show()
