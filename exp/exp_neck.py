from copy import deepcopy
from pandas import DataFrame

from src.data.get_data import get_data
from src.model.make_model import make_model
from src.optim.run_optimizer import run_optim, check_stop
from pipeline.save_load import load_exp, save_reload_comput

nb_iter_per_log = 200


def train_incrementally(data_cfg, model_cfg, optim_cfg, nb_log_per_interval=20):
    '''
    Wrap the train function in a step by step optimization where the results are saved regularly
    '''
    print('WHOLE EXP\ndata_cfg {0} \nmodel_cfg {1} \noptim_cfg {2}'.format(data_cfg, model_cfg, optim_cfg))
    iteration_interval = nb_log_per_interval*nb_iter_per_log
    output, aux_vars, log = load_exp(dict(data_cfg=data_cfg, model_cfg=model_cfg, optim_cfg=optim_cfg))
    exp_done = output is not None
    if not exp_done:
        iteration = 0
        while iteration < optim_cfg['max_iter']:
            temp_optim_cfg = deepcopy(optim_cfg)
            temp_optim_cfg['max_iter'] = min(iteration + iteration_interval, optim_cfg['max_iter'])
            output, aux_vars, log = save_reload_comput(train, check_exp_done, reload_param='max_iter',
                                                       data_cfg=data_cfg, model_cfg=model_cfg, optim_cfg=temp_optim_cfg)
            iteration = iteration + iteration_interval
    return output, aux_vars, log


def train(data_cfg, model_cfg, optim_cfg,
          input=None, aux_vars=None, log=None):
    train_data, test_data = get_data(**data_cfg)

    loss, regularization, net = make_model(train_data, **model_cfg)

    output, aux_vars, log = run_optim(loss, regularization, net, train_data, test_data, **optim_cfg,
                                      nb_iter_per_log=nb_iter_per_log,
                                      input=input, aux_vars=aux_vars, log=log)
    return output, aux_vars, log


def check_exp_done(info_exp):
    return check_stop(info_exp) is not None


def run_exp(data_cfg, model_cfg, optim_cfgs, time=False, add_lr=False):
    info_exp = DataFrame()
    for i, optim_cfg in enumerate(optim_cfgs):
        _, _, log = train_incrementally(data_cfg, model_cfg, optim_cfg)
        algo_name = optim_cfg['oracle']
        if 'algo' in optim_cfg.keys():
            algo_name = algo_name + optim_cfg['algo']
        if 'diff_mode' in optim_cfg.keys():
            algo_name = algo_name + '_' + optim_cfg['diff_mode']
        if add_lr:
            algo_name = algo_name + '_{:1.0e}'.format(optim_cfg['lr_target'])
        if algo_name == 'target_optim':
            algo_name += '_linearized'
        log.update(algo=[algo_name] * len(log['iteration']))

        if time:
            if optim_cfg['oracle'] == 'grad':
                log['iteration'] = [iter*0.3 for iter in log['iteration']]
            else:
                log['iteration'] = [iter*4 for iter in log['iteration']]
        info_exp = info_exp.append(DataFrame(log), ignore_index=True)
    if time:
        info_exp.rename(columns={'iteration': 'time'}, inplace=True)

    return info_exp
