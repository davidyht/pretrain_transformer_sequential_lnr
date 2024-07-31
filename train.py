import torch.multiprocessing as mp
import torch.nn.functional as F
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)  # or 'forkserver'

import argparse
import os
import time

import matplotlib.pyplot as plt
import torch

import numpy as np
import common_args
import random
from dataset import Dataset
from net import Transformer
from utils import (
    build_data_filename,
    build_model_filename,
    convert_to_tensor
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_checkpoints(model, filename):
    # Check if there are existing model checkpoints
    checkpoint_files = [f for f in os.listdir('models') if f.startswith(filename) and 'epoch' in f]
    if checkpoint_files:
        # Sort the checkpoint files based on the epoch number
        checkpoint_files.sort(key=lambda x: int(x.split('_epoch')[1].split('.pt')[0]))
        
        # Get the last checkpoint file
        last_checkpoint_file = checkpoint_files[-1]
        
        # Extract the epoch number from the checkpoint file name
        last_epoch = int(last_checkpoint_file.split('_epoch')[1].split('.pt')[0])
        
        # Load the model checkpoint
        model.load_state_dict(torch.load(os.path.join('models', last_checkpoint_file)))
        
        # Update the starting epoch
        start_epoch = last_epoch + 1
    else:
        # No existing model checkpoints, start from epoch 0
        start_epoch = 0
    return start_epoch, model

def compute_weight(model, batch, batch_size):
    weights = []
    pred_action = model(batch).detach().numpy()

    for i in range(batch_size):
        weight_vec = np.inner(pred_action[i], batch['context_actions'][i])
        weight_vec = np.diagonal(weight_vec)
        weight_vec = np.cumprod(weight_vec)
        weight_vec = weight_vec / weight_vec.sum()  # 归一化处理
        random_vec = np.random.rand(len(weight_vec))
        weight_vec = (random_vec < weight_vec).astype(int)  # 生成所需的向量
        weights.append(weight_vec)
    
    weights = np.array(weights).reshape(-1,)
    return weights



def train():
    if __name__ == '__main__':
        if not os.path.exists('figs/loss'):
            os.makedirs('figs/loss', exist_ok=True)
        if not os.path.exists('models'):
            os.makedirs('models', exist_ok=True)
        parser = argparse.ArgumentParser()
        common_args.add_dataset_args(parser)
        common_args.add_model_args(parser)
        common_args.add_train_args(parser)
        parser.add_argument('--seed', type=int, default=0)
        args = vars(parser.parse_args())
        print("Args: ", args)
        env = args['env']
        n_envs = args['envs']
        n_hists = args['hists']
        n_samples = args['samples']
        horizon = args['H']
        dim = args['dim']
        state_dim = 1
        action_dim = dim
        n_embd = args['embd']
        n_head = args['head']
        n_layer = args['layer']
        context_len = args['context_len']
        lr = args['lr']
        shuffle = args['shuffle']
        dropout = args['dropout']
        var = args['var']
        cov = args['cov']
        num_epochs = args['num_epochs']
        seed = args['seed']
        tmp_seed = seed
        if seed == -1:
            tmp_seed = 0
        torch.manual_seed(tmp_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(tmp_seed)
            torch.cuda.manual_seed_all(tmp_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(tmp_seed)
        random.seed(tmp_seed)
       
        dataset_config = {
            'n_hists': n_hists,
            'n_samples': n_samples,
            'horizon': horizon,
            'dim': dim,
        }
        dataset_config.update({'var': var, 'cov': cov, 'type': 'uniform'})

        model_config = {
            'shuffle': shuffle,
            'lr': lr,
            'dropout': dropout,
            'n_embd': n_embd,
            'n_layer': n_layer,
            'n_head': n_head,
            'n_envs': n_envs,
            'n_hists': n_hists,
            'n_samples': n_samples,
            'horizon': horizon,
            'dim': dim,
            'seed': seed,
            'context_len': context_len,
        }
        model_config.update({'var': var, 'cov': cov})
        config = {
            'horizon': horizon,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'n_layer': n_layer,
            'n_embd': n_embd,
            'n_head': n_head,
            'shuffle': shuffle,
            'dropout': dropout,
            'test': False,
            'store_gpu': True,
            'context_len': context_len,
        }

        model = Transformer(config).to(device)
        params = {
            'batch_size': 100,
            'shuffle': True,
            'drop_last': True,
        }
        batch_size = 100
        filename = build_model_filename(env, model_config)
        log_filename = f'figs/loss/{filename}_logs.txt'
        with open(log_filename, 'w') as f:
            pass
        def printw(string):
            """
            A drop-in replacement for print that also writes to a log file.
            """
            # Use the standard print function to print to the console
            print(string)
            # Write the same output to the log file
            with open(log_filename, 'a') as f:
                f.write(string + '\n')
        path_train = build_data_filename(env, n_envs, dataset_config, mode=0)
        path_test = build_data_filename(env, n_envs, dataset_config, mode=1)
        train_dataset = Dataset(path = path_train, config = config)
        test_dataset = Dataset(path = path_test, config = config)
        train_loader = torch.utils.data.DataLoader(train_dataset, **params)
        test_loader = torch.utils.data.DataLoader(test_dataset, **params)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        test_loss = []
        train_loss = []
        printw("Num train batches: " + str(len(train_loader)))
        printw("Num test batches: " + str(len(test_loader)))
        start_epoch, model = load_checkpoints(model, filename)
        if start_epoch == 0:
            printw("Starting from scratch.")
        else:
            printw(f"Starting from epoch {start_epoch}")

        for epoch in range(start_epoch, num_epochs):
            # EVALUATION
            printw(f"Epoch: {epoch + 1}")
            start_time = time.time()
            with torch.no_grad():
                epoch_test_loss = 0.0
                for i, batch in enumerate(test_loader):
                    print(f"Batch {i} of {len(test_loader)}", end='\r')

                    batch = {k: v.to(device) for k, v in batch.items()}
                    true_actions = batch['true_actions']

                    pred_actions = model(batch)
                    true_actions = true_actions.reshape(-1, action_dim)
                    pred_actions = pred_actions.reshape(-1, action_dim)
                
                    loss = loss_fn(pred_actions, true_actions)
                    epoch_test_loss += loss.item() / horizon
            test_loss.append(epoch_test_loss / len(test_dataset))
            end_time = time.time()
            printw(f"\tTest loss: {test_loss[-1]}")
            printw(f"\tEval time: {end_time - start_time}")
            # TRAINING
            epoch_train_loss = 0.0
            start_time = time.time()
            
            for i, batch in enumerate(train_loader):
                weights = compute_weight(model, batch, batch_size)
                weights = torch.tensor(weights, dtype=torch.float32).detach().to(device)
                print(f"Batch {i} of {len(train_loader)}", end='\r')
                batch = {k: v.to(device) for k, v in batch.items()}

                true_actions = batch['true_actions'].reshape(-1, action_dim)
                pred_actions = model(batch)
                pred_actions = pred_actions.reshape(-1, action_dim)
                optimizer.zero_grad()
                loss = F.cross_entropy(pred_actions, true_actions, reduction='none')
                weighted_loss = torch.inner(loss, weights)
                weighted_loss.backward()
                optimizer.step()
                loss = loss.mean()

                epoch_train_loss += loss.item() / horizon
            train_loss.append(epoch_train_loss / len(train_loader.dataset))
            end_time = time.time()
            printw(f"\tTrain loss: {train_loss[-1]}")
            printw(f"\tTrain time: {end_time - start_time}")
            # LOGGING
            if (epoch + 1) % 20 == 0:
                torch.save(model.state_dict(), f'models/{filename}_epoch{epoch+1}.pt')

            # PLOTTING
            if (epoch + 1) % 10 == 0:
                printw(f"Test Loss:        {test_loss[-1]}")
                printw(f"Train Loss:       {train_loss[-1]}")
                printw("\n")
                plt.yscale('log')
                plt.plot(train_loss[1:], label="Train Loss")
                plt.plot(test_loss[1:], label="Test Loss")
                plt.legend()
                plt.savefig(f"figs/loss/{filename}_train_loss.png")
                plt.clf()
        torch.save(model.state_dict(), f'models/{filename}.pt')
        print("Done.")


train()