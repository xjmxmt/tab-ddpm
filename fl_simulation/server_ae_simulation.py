import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
sys.path.append(r'C:\Users\xujia\Desktop\DiffusionModel\code\EntityEncoder')
sys.path.append(r'/mnt/c/Users/xujia/Desktop/DiffusionModel/code/EntityEncoder')
import tomli
import shutil
import argparse
import tensorflow as tf
import torch
import torch.utils.data
import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict
import flwr as fl
from flwr.server.client_proxy import ClientProxy, FitRes
from flwr.common import Parameters, Scalar
from typing import List, Tuple, Union, Optional, Dict
from multiprocessing import Process
from lib.logger import Logger
from lib.util import tensor2ndarray
from entityencoder_utils.data_utils import get_dataset
from entity_encoders.encoder import Encoder
from scripts.sample import sample
from scripts.eval_catboost import train_catboost
from util import dirichlet_split_noniid
# from client_ae_simulation import EncoderClient
# from client_simulation import DiffusionClient


def load_config(path):
    with open(path, 'rb') as f:
        return tomli.load(f)


def save_file(parent_dir, config_path):
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "current_round": server_round,
    }
    return config


class PretrainingSaveModelStrategy(fl.server.strategy.FedAvg):
    def set_aggregation_weights(self, weights: dict):
        self.weights = weights

    def get_aggregation_weights(self):
        print('aggregation weights: ', self.weights)
        return self.weights

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # if not results:
        #     return None, {}
        # # Do not aggregate if there are failures and failures are not accepted
        # if not self.accept_failures and failures:
        #     return None, {}
        #
        # # Convert results
        # weights_results = [
        #     (fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        #     # (fl.common.parameters_to_ndarrays(fit_res.parameters), self.weights[client.cid])
        #     for client, fit_res in results
        # ]
        # aggregated_parameters = fl.common.ndarrays_to_parameters(fl.server.strategy.aggregate.aggregate(weights_results))
        #
        # # Aggregate custom metrics if aggregation fn was provided
        # aggregated_metrics = {}
        # if self.fit_metrics_aggregation_fn:
        #     fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        #     aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        # elif server_round == 1:  # Only log this warning once
        #     print("No fit_metrics_aggregation_fn provided")

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            if server_round == pretraining_rounds and not os.path.exists(os.path.join(parent_dir, "ae_weights.npz")):
                # Save aggregated_ndarrays
                print(f'Saving round {server_round} aggregated ndarrays.')
                np.savez(os.path.join(parent_dir, "ae_weights.npz"), *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics


parser = argparse.ArgumentParser()
parser.add_argument('--config', metavar='FILE')
args = parser.parse_args()

# config_path = sys.argv[-1]
# config_path = '../exp/adult/configs/config-ae-cat-fl.toml'
config_path = args.config
print('config path: ', config_path)

raw_config = load_config(config_path)
parent_dir = raw_config['parent_dir']
# ds_name = raw_config['ds_name']
# real_data_path = raw_config['real_data_path']
# device = raw_config['device']

save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), config_path)

num_clients = raw_config['fl_params']['num_clients']
num_rounds = raw_config['fl_params']['num_rounds']
pretraining_rounds = raw_config['fl_params']['pretraining_rounds']

# sys.stdout = Logger(parent_dir=raw_config['parent_dir'],
#                     ds_name=raw_config['ds_name'],
#                     encoding_type=raw_config['encoding_type'],
#                     decoding_type=raw_config['decoding_type'],
#                     stream=sys.stdout)
# # sys.stdout.fileno = lambda: False
#
# sys.stderr = Logger(parent_dir=raw_config['parent_dir'],
#                     ds_name=raw_config['ds_name'],
#                     encoding_type=raw_config['encoding_type'],
#                     decoding_type=raw_config['decoding_type'],
#                     stream=sys.stderr)
# # sys.stderr.fileno = lambda: False

# data, preprocessors, processed_dataset_info, _ = \
#     get_dataset(raw_config['real_data_path'], raw_config['ds_info'], raw_config['parent_dir'],
#                 encoding=raw_config['encoding_type'], decoding=raw_config['decoding_type'],
#                 is_y_cond=raw_config['model_params']['is_y_cond'], auto_search=False, return_encoder=False,
#                 seed=raw_config['seed'], device=device)
#
# X_train, y_train = data['X_train'], data['y_train']
# X_val, y_val = data['X_val'], data['y_val']
# X_test, y_test = data['X_test_tensor'], data['y_test_tensor']
# train_length = len(X_train)
# eval_length = len(X_val)
# test_length = len(X_test)
#
# dataloaders_dict = {}
# batch_size = raw_config['train']['main']['batch_size']
# if raw_config['ds_split'] == 'average' or raw_config['ds_split'] == 'uneven':
#     split_type = raw_config['ds_split']
#     train_split_dict = get_data_splits(train_length, num_clients, type=split_type)
#     eval_split_dict = get_data_splits(eval_length, num_clients, type=split_type)
#     test_split_dict = get_data_splits(test_length, num_clients, type=split_type)
#     print(train_split_dict, eval_split_dict, test_split_dict)
#
#     for i in range(num_clients):
#         cid = str(i)
#         X_train_split = X_train[train_split_dict[cid][0]:train_split_dict[cid][1]]
#         y_train_split = y_train[train_split_dict[cid][0]:train_split_dict[cid][1]]
#         train_dataset = torch.utils.data.TensorDataset(X_train_split, y_train_split)
#         train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         X_val_split = X_val[eval_split_dict[cid][0]:eval_split_dict[cid][1]]
#         y_val_split = y_val[eval_split_dict[cid][0]:eval_split_dict[cid][1]]
#         eval_dataset = torch.utils.data.TensorDataset(X_val_split, y_val_split)
#         eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
#         X_test_split = X_test[test_split_dict[cid][0]:test_split_dict[cid][1]]
#         y_test_split = y_test[test_split_dict[cid][0]:test_split_dict[cid][1]]
#         test_dataset = torch.utils.data.TensorDataset(X_test_split, y_test_split)
#         test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#         dataloaders = {
#             'train': train_loader,
#             'eval': eval_loader,
#             'test': test_loader
#         }
#         dataloaders_dict[cid] = dataloaders
#
# elif raw_config['ds_split'] == 'dirichlet':
#     alpha = raw_config['alpha']
#     train_client_idcs = dirichlet_split_noniid(train_labels=y_train, n_clients=num_clients, alpha=alpha, seed=raw_config['seed'])
#     eval_client_idcs = dirichlet_split_noniid(train_labels=y_val, n_clients=num_clients, alpha=alpha, seed=raw_config['seed'])
#     test_client_idcs = dirichlet_split_noniid(train_labels=y_test, n_clients=num_clients, alpha=alpha, seed=raw_config['seed'])
#
#     for i in range(num_clients):
#         cid = str(i)
#         print(cid, len(train_client_idcs[i]))
#         X_train_split = X_train[train_client_idcs[i]]
#         y_train_split = y_train[train_client_idcs[i]]
#         train_dataset = torch.utils.data.TensorDataset(X_train_split, y_train_split)
#         train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         X_val_split = X_val[eval_client_idcs[i]]
#         y_val_split = y_val[eval_client_idcs[i]]
#         eval_dataset = torch.utils.data.TensorDataset(X_val_split, y_val_split)
#         eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
#         X_test_split = X_test[test_client_idcs[i]]
#         y_test_split = y_test[test_client_idcs[i]]
#         test_dataset = torch.utils.data.TensorDataset(X_test_split, y_test_split)
#         test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#         dataloaders = {
#             'train': train_loader,
#             'eval': eval_loader,
#             'test': test_loader
#         }
#         dataloaders_dict[cid] = dataloaders

# os.makedirs(parent_dir, exist_ok=True)
# encoder_clients = []
# diffusion_clients = []
# for i in range(num_clients):
#     client = EncoderClient(f'--config {config_path} --train --cid {i}', processed_dataset_info, dataloaders_dict[str(i)])
#     encoder_clients.append(client)

# Create FedAvg strategy
strategy = PretrainingSaveModelStrategy(
        fraction_fit=1.0,  # Sample % of available clients for training
        fraction_evaluate=1.0,  # Sample % of available clients for evaluation
        min_fit_clients=2,  # Never sample less than n clients for training
        min_evaluate_clients=2,  # Never sample less than n clients for evaluation
        min_available_clients=num_clients,  # Wait until all n clients are available
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config
)

# fl.simulation.start_simulation(
#     client_fn=lambda x: encoder_clients[int(x)],
#     num_clients=num_clients,
#     config=fl.server.ServerConfig(num_rounds=pretraining_rounds),
#     strategy=strategy,
#     client_resources={"num_gpus": 1},
#     ray_init_args={
#         "ignore_reinit_error": True,
#         "include_dashboard": False,
#         "num_gpus": 1
#     },
# )

fl.server.start_server(
    server_address="0.0.0.0:9999",
    config=fl.server.ServerConfig(num_rounds=pretraining_rounds),
    strategy=strategy
)
