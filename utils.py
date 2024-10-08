import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**32) + worker_id
    torch.manual_seed(worker_seed)
    numpy_seed = int(worker_seed % (2**32 - 1))  # Optional, in case you also use numpy in the DataLoader
    np.random.seed(numpy_seed)


def build_data_filename(env, n_envs, config, mode):
    """
    Builds the filename for the bandit data.
    Mode is either 0: train, 1: test, 2: eval.
    """
    filename_template = 'datasets/trajs_{}.pkl'
    filename = env
    filename += '_envs' + str(n_envs)
    if mode != 2:
        filename += '_hists' + str(config['n_hists'])
        filename += '_samples' + str(config['n_samples'])
    filename += '_H' + str(config['horizon'])
    filename += '_d' + str(config['dim'])
    filename += '_var' + str(config['var'])
    filename += '_cov' + str(config['cov'])
    if mode == 0:
        filename += '_train'
    elif mode == 1:
        filename += '_test'
    elif mode == 2:
        filename += '_eval'
    return filename_template.format(filename)


def build_model_filename(env, config):
    """
    Builds the filename for the cgbandit model.
    """
    filename = env
    filename += '_shuf' + str(config['shuffle'])
    filename += '_lr' + str(config['lr'])
    filename += '_do' + str(config['dropout'])
    filename += '_embd' + str(config['n_embd'])
    filename += '_layer' + str(config['n_layer'])
    filename += '_head' + str(config['n_head'])
    filename += '_envs' + str(config['n_envs'])
    filename += '_hists' + str(config['n_hists'])
    filename += '_samples' + str(config['n_samples'])
    filename += '_var' + str(config['var'])
    filename += '_cov' + str(config['cov'])
    filename += '_H' + str(config['horizon'])
    filename += '_d' + str(config['dim'])
    filename += '_seed' + str(config['seed'])
    return filename


def build_merge_data_filename(env_names, n_envs, config, mode):
    """
    Builds the filename for the merged bandit data.
    Mode is either 0: train, 1: test, 2: eval.
    """
    filename_template = 'datasets/trajs_{}.pkl'
    filename = '_'.join(env_names)
    filename += '_envs' + str(n_envs)
    if mode != 2:
        filename += '_hists' + str(config['n_hists'])
        filename += '_samples' + str(config['n_samples'])
    filename += '_H' + str(config['horizon'])
    filename += '_d' + str(config['dim'])
    filename += '_var' + str(config['var'])
    filename += '_cov' + str(config['cov'])
    if mode == 0:
        filename += '_train'
    elif mode == 1:
        filename += '_test'
    elif mode == 2:
        filename += '_eval'
    return filename_template.format(filename)

def build_merge_model_filename(env_names, config):
    """
    Builds the filename for the merged bandit model.
    """
    filename = '_'.join(env_names)
    filename += '_shuf' + str(config['shuffle'])
    filename += '_lr' + str(config['lr'])
    filename += '_do' + str(config['dropout'])
    filename += '_embd' + str(config['n_embd'])
    filename += '_layer' + str(config['n_layer'])
    filename += '_head' + str(config['n_head'])
    filename += '_envs' + str(config['n_envs'])
    filename += '_hists' + str(config['n_hists'])
    filename += '_samples' + str(config['n_samples'])
    filename += '_var' + str(config['var'])
    filename += '_cov' + str(config['cov'])
    filename += '_H' + str(config['horizon'])
    filename += '_d' + str(config['dim'])
    filename += '_seed' + str(config['seed'])
    return filename

def convert_to_tensor(x, store_gpu=True):
    if store_gpu:
        return torch.tensor(np.asarray(x)).float().to(device)
    else:
        return torch.tensor(np.asarray(x)).float()