import numpy as np
from pipeline.save_load import get_exp_folder, save_grid_search, load_grid_search
from pipeline.utils_grid import build_list_from_grid, build_grid_from_cfg, set_cfg
CRED = '\033[91m'
CEND = '\033[0m'


def grid_search(exp_cfg, method, check_stop, criterion, criterion_measure='min_area'):
    """
    Gird search on the experiment configuration exp_cfg for the given method.
    Args:
        exp_cfg: (dict of dicts) such as exp_cfg=(data_cfg=..., model_cfg=..., optim_cfg=...)
                (each entry being a dictionary)
                some entries of these dictionaries must be lists that are used to define a grid of parameters,
                this grid of parameters is transformed in a list of all possible combinations,
                the best parameters for the grid are returned
        method: (function) function to run the experiment, e.g. train (see exp_neck)
        check_stop: (function) function to check if the experiment diverged, (see exp_neck)
        criterion: (str) what is the relevant information to look at from the experiment
                    to rank the different configurations and extract the best one, e.g. 'train_loss'
        criterion_measure: (str) method to measure the criterion such as the minimal area under the curve

    Returns:
        best_params: (dict) best parameters from the varying parameters in the exp_cfg
                            (defined by the entries of the dictionary that are lists)
                            So typically if optim_cfg had two entries lr=[...], reg=[...] that were lists,
                            all possible combinations of those two lists were run and the best_params would be
                            best_params = dict(lr=best_lr, reg=best_reg)
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
                if 'area' in criterion_measure:
                    measure = np.trapz(info_exp[criterion], info_exp['iteration'])
                else:
                    raise NotImplementedError

            best_measure_cfg.append(measure)

        if 'min' in criterion_measure:
            idx_best = int(np.argmin(np.array(best_measure_cfg)))
        elif 'max' in criterion_measure:
            idx_best = int(np.argmax(np.array(best_measure_cfg)))
        else:
            raise NotImplementedError
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



