from copy import deepcopy


def build_list_from_grid(params_grid):
    """
    Create a list of parameters from a grid of any size
    :param params_grid: (dict) dictionary containing parameters name and their range on which the grid search is done.
                    e.g. params_grid = dict(step_size = [1,2,3], line_search=['armijo', 'wolfe'])
    :return:
        params_list: (list) list of all possible configurations of the parameters given in the grid,
                    e.g. params_list[0] = dict(step_size=1, line_search='armijo')
    """
    param_sample0 = {key: None for key in params_grid.keys()}
    params_list = [param_sample0]
    for param_name, param_range in params_grid.items():
        new_params_list = []
        for param_sample in params_list:
            for param in param_range:
                new_param_sample = deepcopy(param_sample)
                new_param_sample[param_name] = param
                new_params_list.append(new_param_sample)
        params_list = deepcopy(new_params_list)
    return params_list


def build_grid_from_cfg(exp_cfg):
    """
    Scan the exp_cfg and extract entries that are lists to be searched on by e.g. a grid-search
    Args:
        exp_cfg: (dict of dicts) such as exp_cfg=(data_cfg=..., model_cfg=..., optim_cfg=...)
                (each entry being a dictionary)
                some entries of these dictionaries must be lists that are used to define a grid of parameters

    Returns:
        params_grid: (dict) dictionary of the form dict(param1=[], param2=[], ...)
                    where each param_i corresponds to one parameter in the exp_cfg that was given in the form of a list
                    the list is then the corresponding value of param_i in this dictionary

    """
    params_grid = dict()
    for cfg in exp_cfg.values():
        for key, value in cfg.items():
            if isinstance(value, list):
                params_grid.update({key: value})
    return params_grid


def build_list_exp(exp_cfgs):
    """
    Exp_cfgs is a list of exp_cfg = [data_cfg, model_cfg, optim_cfg]
    Each data_cfg, model_cfg, optim_cfg can have parameters that are list (as in grid_search)
    Therefore for each exp_cfg corresponds all combinations of these parameters
    But the combinations of the parameters of two different exp_cfg (in the list exp_cfgs) won't be combined
    Can be useful if for example one wants to run different algorithms with different parameter names
    At the end, it builds a total list of experiments that can be run on a cluster in parallel
    by assigning different portions of the list to each node

    The function is only used to run experiments on a cluster, all results are saved and then used by other functions

    Args:
        exp_cfgs: (list of dicts of dicts) [exp_cfg1, exp_cfg2, ...]
                where each exp_cfg is of the form e.g. exp_cfg_i=(data_cfg=..., model_cfg=..., optim_cfg=...)
                (each entry being a dictionary)
                some entries of these dictionaries can be lists that are used to define a grid of parameters

    Returns:
        params_grid: (dict) dictionary of the form dict(param1=[], param2=[], ...)
                    where each param_i corresponds to one parameter in the exp_cfg that was given in the form of a list
                    the list is then the corresponding value of param_i in this dictionary
    """
    exp_cfgs_list = list()
    for exp_cfg in exp_cfgs:
        params_grid = dict()
        for cfg in exp_cfg.values():
            for key, value in cfg.items():
                if isinstance(value, list):
                    params_grid.update({key: value})
        params_list = build_list_from_grid(params_grid)
        for params in params_list:
            exp_cfg = {key: set_cfg(cfg, params) for key, cfg in exp_cfg.items()}
            exp_cfgs_list.append(exp_cfg)
    return exp_cfgs_list


def set_cfg(default_cfg, given_params):
    """
    Set cfg with the given_params
    :param default_cfg: (dict) one of data_cfg, model_cfg, optim_cfg dictionaries
    :param given_params: (dict) params to include in the default_cfg
                        e.g. given_params = dict(step_size=0, line_search='wolfe')
    :return:
        cfg_to_test: (dict) updated cfg with the given params
    """
    cfg_to_test = deepcopy(default_cfg)
    for param_key in default_cfg.keys():
        if param_key in given_params.keys():
            cfg_to_test[param_key] = given_params[param_key]
    return cfg_to_test