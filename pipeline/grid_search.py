import numpy as np
from pipeline.save_load import get_exp_folder, save_grid_search, load_grid_search
from pipeline.utils_grid import build_list_from_grid, build_grid_from_cfg, set_cfg
CRED = '\033[91m'
CEND = '\033[0m'


def grid_search(exp_cfg, method, check_stop, criterion):
    """
    Gird search on the experiment configuration exp_cfg for the given method.
    Args:
        exp_cfg: (dict of dicts) such as data_cfg, model_cfg, optim_cfg (each being a dictionary)
                some entries of these dictionaries must be lists that are used to define a grid of parameters,
                this grid of parameters is transformed in a list of all possible combinations,
                the best parameters for the grid are returned
        method: (function) function to run the experiment, e.g. train (see exp_neck)
        check_stop: (function) function to check if the experiment diverged, (see e.g. exp_neck)
        criterion: (str) what should be measured from the information logged in the experiment, e.g. 'train_loss'

    Returns:
        best_params: (dict) best parameters from the varying parameters in the exp_cfg
                            (defined by the entries of the dictionary that are lists)

    """
    best_params = load_grid_search(exp_cfg)
    if best_params is None:
        params_grid = build_grid_from_cfg(exp_cfg)
        assert len(params_grid) > 0

        params_list = build_list_from_grid(params_grid)
        best_measure_cfg = list()
        for i, params in enumerate(params_list):
            search_exp_cfg = {key: set_cfg(cfg, params) for key, cfg in exp_cfg.items()}
            print(CRED + 'Grid search percentage {0:2.0f}'.format(float(i/len(params_list))*100) + CEND)
            _, _, info_exp = method(**search_exp_cfg)
            stopped = check_stop(info_exp)
            if stopped == 'diverged':
                measure = 10 ** 10
            else:
                measure = np.trapz(info_exp[criterion], info_exp['iteration'])

            best_measure_cfg.append(measure)
        idx_best = int(np.argmin(np.array(best_measure_cfg)))
        best_params = params_list[idx_best]

        save_grid_search(exp_cfg, best_params)

        file_path = get_exp_folder() + '/results/grid_search_records.txt'
        with open(file_path, 'a') as file:
            to_write = ''
            for cfg in exp_cfg:
                to_write += '{0}\n'.format(str(cfg))
            to_write += 'best params: ' + str(best_params) + '\n\n'
            file.write(to_write)
    print(CRED + 'best params:' + str(best_params) + CEND)
    return best_params



