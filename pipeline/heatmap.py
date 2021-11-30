import seaborn as sns
import os
from numpy import trapz
import math
from pandas import DataFrame
from matplotlib import pyplot as plt

from pipeline.utils_grid import build_list_from_grid, build_grid_from_cfg, set_cfg
from pipeline.save_load import get_exp_folder, save, load, format_files


def compute_heatmap(exp_cfg, x_param_name, y_param_name, method, check_exp, criterion):
    """
    Heatmap that runs and displays the measures
    Args:
        exp_cfg: (dict of dicts) the inner dicts are e.g. data_cfg, model_cfg, optim_cfg
                some entries of these dictionaries must be lists that are used to define a grid of parameters,
        x_param_name: (str) key of the first parameter varying in one of the dict of exp_cfg
        y_param_name: (str) key of the second parameter varying in one of the dict of exp_cfg
        method: (function) function to run the experiment, e.g. train (see exp_neck)
        check_exp: (function) function to check if the experiment diverged, (see e.g. exp_neck)
        criterion: (str) what should be measured from the information logged in the experiment, e.g. 'train_loss'

    Returns:
        heatmap: (dict of lists) dictionary containing the measures estimated by the heatmap
                                 for each combination of parameters

    """
    exp_folder = get_exp_folder()
    heatmap_path = '{0}/heatmaps/{1}_{2}{3}'.format(exp_folder, x_param_name, y_param_name, format_files)
    if not os.path.exists(heatmap_path):
        params_grid = build_grid_from_cfg(exp_cfg)
        for key in params_grid.keys():
            if key not in [x_param_name, y_param_name]:
                raise ValueError

        x_params = params_grid[x_param_name]
        y_params = params_grid[y_param_name]

        heatmap = {x_param_name: list(), y_param_name: list(), 'measure': list()}
        counter = 0
        for x_param in x_params:
            for y_param in y_params:
                params = {x_param_name: x_param, y_param_name: y_param}
                search_exp_cfg = {key: set_cfg(cfg, params) for key, cfg in exp_cfg.items()}
                print(*['{0}:{1}'.format(key, value) for key, value in search_exp_cfg.items()], sep='\n')

                _, _, info_exp = method(**search_exp_cfg)
                stopped = check_exp(info_exp)
                if stopped == 'has diverged':
                    measure = float('nan')
                else:
                    measure = trapz(info_exp[criterion], info_exp['iteration'])
                print(measure)
                print(info_exp[criterion][-1])
                heatmap[x_param_name].append(x_param)
                heatmap[y_param_name].append(y_param)
                heatmap['measure'].append(1/measure)
                print('Heatmap percentage {0:2.0f}'.format(float(counter/len(build_list_from_grid(params_grid)))*100))
                counter += 1
        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
        with open(heatmap_path, 'wb') as file:
            save(heatmap, file)
    else:
        with open(heatmap_path, 'rb') as file:
            heatmap = load(file)
    return heatmap


def plot_heatmap(heatmap, x_param_name, y_param_name):
    heatmap = DataFrame(heatmap)
    heatmap = heatmap.pivot(x_param_name, y_param_name, 'measure')
    ax = sns.heatmap(heatmap)
    fig = ax.get_figure()
    fig.canvas.draw()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    return ax


