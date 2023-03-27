import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

sys.path.append(r'C:\Users\xujia\Desktop\DiffusionModel\code\EntityEncoder')
sys.path.append(r'/mnt/c/Users/xujia/Desktop/DiffusionModel/code/EntityEncoder')

import tomli
import shutil
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
from lib.logger import Logger
from lib.util import tensor2ndarray
from entityencoder_utils.data_utils import get_dataset
from entity_encoders.encoder import Encoder
from scripts.sample import sample
from scripts.eval_catboost import train_catboost
from util import dirichlet_split_noniid
from client_ae_simulation import EncoderClient
from client_simulation import DiffusionClient


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


def get_data_splits(dataset_length, num_clients, type='uneven'):
    data_split_dict = {}
    if type == 'average':
        # average split
        split = dataset_length // num_clients
        start = 0
        for i in range(num_clients-1):
            data_split_dict[str(i)] = (start, start + split)
            start += split
        data_split_dict[str(num_clients-1)] = (start, dataset_length)
    elif type == 'uneven':
        splits = np.random.randint(dataset_length // num_clients // 4, dataset_length // num_clients * 4, num_clients)

        while sum(splits) > dataset_length:
            splits = np.random.randint(dataset_length // num_clients // 4, dataset_length // num_clients * 4, num_clients)
        if sum(splits) < dataset_length:
            left = dataset_length - sum(splits)
            splits += left // num_clients

        start = 0
        for i in range(num_clients-1):
            data_split_dict[str(i)] = (start, start + splits[i])
            start += splits[i]
        data_split_dict[str(num_clients - 1)] = (start, dataset_length)
    return data_split_dict


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


class SaveModelStrategy(fl.server.strategy.FedAvg):

    def set_aggregation_weights(self, weights: dict):
        self.weights = weights

    def get_aggregation_weights(self):
        print('aggregation weights: ', self.weights)
        return self.weights

    def set_report_history(self, is_regression):
        if is_regression:
            self.report_history = pd.DataFrame(columns=['round', 'r2', 'rmse', 'mape', 'evs', 'disc f1', 'avg_jsd', 'avg_wd', 'correlation_diff'])
        else:
            self.report_history = pd.DataFrame(columns=['round', 'acc', 'f1', 'roc_auc', 'disc f1', 'avg_jsd', 'avg_wd', 'correlation_diff'])

    def get_report_history(self):
        print('aggregated loss history:', self.report_history)
        return self.report_history

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        # aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            # (fl.common.parameters_to_ndarrays(fit_res.parameters), self.weights[client.cid])
            for client, fit_res in results
        ]
        aggregated_parameters = fl.common.ndarrays_to_parameters(fl.server.strategy.aggregate.aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        aggregated_metrics = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            print("No fit_metrics_aggregation_fn provided")

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # if server_round == num_rounds and not os.path.exists(os.path.join(parent_dir, "model.npz")):
            #     # Save aggregated_ndarrays
            #     print(f'Saving round {server_round} aggregated ndarrays.')
            #     np.savez(os.path.join(parent_dir, "model.npz"), *aggregated_ndarrays)

            print(f'Saving round {server_round} aggregated ndarrays.')
            np.savez(os.path.join(parent_dir, f"model-rnd-{str(server_round)}.npz"), *aggregated_ndarrays)

            synthetic_data = sample(
                data,
                preprocessors,
                processed_dataset_info,
                aggregated_entity_encoder,
                num_samples=raw_config['sample']['num_samples'],
                batch_size=raw_config['sample']['batch_size'],
                disbalance=raw_config['sample'].get('disbalance', None),
                **raw_config['diffusion_params'],
                parent_dir=raw_config['parent_dir'],
                model_path=os.path.join(raw_config['parent_dir'], f"model-rnd-{str(server_round)}.npz"),
                model_type=raw_config['model_type'],
                model_params=raw_config['model_params'],
                device=device,
                seed=raw_config['sample'].get('seed', 0),
                # change_val=args.change_val
            )

            report = train_catboost(
                raw_data=data,
                dataset_info=processed_dataset_info,
                ds_name=raw_config['ds_name'],
                parent_dir=raw_config['parent_dir'],
                preprocessors=preprocessors,
                seed=raw_config['seed'],
                eval_type=raw_config['eval']['type']['eval_type'],
                synthetic_data=synthetic_data,
                get_discriminator_measure=True,
                get_statistical_similarity=True
            )

            self.report = report

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self, rnd: int, results, failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        metrics_averaged = super().aggregate_evaluate(rnd, results, failures)[0]

        # writer_distributed = tf.summary.create_file_writer("logs/distributed")
        #
        # if rnd != -1:
        #     step = rnd
        # else:
        #     step = len(
        #         [
        #             name
        #             for name in os.listdir("logs/distributed")
        #             if os.path.isfile(os.path.join("logs/distributed", name))
        #         ]
        #     )
        #
        # with writer_distributed.as_default():
        #     for client_idx, (_, evaluate_res) in enumerate(results):
        #         tf.summary.scalar(
        #             f"num_examples_client_{client_idx + 1}",
        #             evaluate_res.num_examples,
        #             step=step,
        #         )
        #
        #         if processed_dataset_info['is_regression']:
        #             tf.summary.scalar(
        #                 f"r2_client_{client_idx + 1}", evaluate_res.metrics['report']['ml efficiency']['test']['r2'], step=step
        #             )
        #             tf.summary.scalar(
        #                 f"rmse_client_{client_idx + 1}", evaluate_res.metrics['report']['ml efficiency']['test']['rmse'], step=step
        #             )
        #             tf.summary.scalar(
        #                 f"mape_client_{client_idx + 1}", evaluate_res.metrics['report']['ml efficiency']['test']['mape'], step=step
        #             )
        #             tf.summary.scalar(
        #                 f"evs_client_{client_idx + 1}", evaluate_res.metrics['report']['ml efficiency']['test']['evs'], step=step
        #             )
        #         else:
        #             tf.summary.scalar(
        #                 f"acc_client_{client_idx + 1}", evaluate_res.metrics['report']['ml efficiency']['test']['acc'], step=step
        #             )
        #             tf.summary.scalar(
        #                 f"f1_client_{client_idx + 1}", evaluate_res.metrics['report']['ml efficiency']['test']['f1'], step=step
        #             )
        #             tf.summary.scalar(
        #                 f"roc_auc_client_{client_idx + 1}", evaluate_res.metrics['report']['ml efficiency']['test']['roc_auc'], step=step
        #             )
        #
        #         tf.summary.scalar(
        #             f"discriminator_client_{client_idx + 1}", evaluate_res.metrics['report']['discriminator measure']['test']['macro f1'], step=step
        #         )
        #         tf.summary.scalar(
        #             f"avg_jsd_client_{client_idx + 1}", evaluate_res.metrics['report']['statistical similarity']['avg_jsd'], step=step
        #         )
        #         tf.summary.scalar(
        #             f"avg_wd_client_{client_idx + 1}", evaluate_res.metrics['report']['statistical similarity']['avg_wd'], step=step
        #         )
        #         tf.summary.scalar(
        #             f"correlation_diff_client_{client_idx + 1}", evaluate_res.metrics['report']['statistical similarity']['correlation_diff'], step=step
        #         )
        #
        #     writer_distributed.flush()

        if processed_dataset_info['is_regression']:
            self.report_history.loc[len(self.report_history)] = [rnd,
                                                                 self.report['ml efficiency']['test']['r2'],
                                                                 self.report['ml efficiency']['test']['rmse'],
                                                                 self.report['ml efficiency']['test']['mape'],
                                                                 self.report['ml efficiency']['test']['evs'],
                                                                 self.report['discriminator measure']['test']['macro f1'],
                                                                 self.report['statistical similarity']['avg_jsd'],
                                                                 self.report['statistical similarity']['avg_wd'],
                                                                 self.report['statistical similarity']['correlation_diff']]
        else:
            self.report_history.loc[len(self.report_history)] = [rnd,
                                                                 self.report['ml efficiency']['test']['acc'],
                                                                 self.report['ml efficiency']['test']['f1'],
                                                                 self.report['ml efficiency']['test']['roc_auc'],
                                                                 self.report['discriminator measure']['test']['macro f1'],
                                                                 self.report['statistical similarity']['avg_jsd'],
                                                                 self.report['statistical similarity']['avg_wd'],
                                                                 self.report['statistical similarity']['correlation_diff']]

        # with writer_federated.as_default():
        #     if processed_dataset_info['is_regression']:
        #         tf.summary.scalar(
        #             f"r2_aggregator", self.report['ml efficiency']['test']['r2'], step=step
        #         )
        #         tf.summary.scalar(
        #             f"rmse_aggregator", self.report['ml efficiency']['test']['rmse'], step=step
        #         )
        #         tf.summary.scalar(
        #             f"mape_aggregator", self.report['ml efficiency']['test']['mape'], step=step
        #         )
        #         tf.summary.scalar(
        #             f"evs_aggregator", self.report['ml efficiency']['test']['evs'], step=step
        #         )
        #     else:
        #         tf.summary.scalar(
        #             f"acc_aggregator", self.report['ml efficiency']['test']['acc'], step=step
        #         )
        #         tf.summary.scalar(
        #             f"f1_aggregator", self.report['ml efficiency']['test']['f1'], step=step
        #         )
        #         tf.summary.scalar(
        #             f"roc_auc_aggregator", self.report['ml efficiency']['test']['roc_auc'], step=step
        #         )
        #
        #     tf.summary.scalar(
        #         f"discriminator_aggregator",
        #         self.report['discriminator measure']['test']['macro f1'], step=step
        #     )
        #     tf.summary.scalar(
        #         f"avg_jsd_aggregator", self.report['statistical similarity']['avg_jsd'],
        #         step=step
        #     )
        #     tf.summary.scalar(
        #         f"avg_wd_aggregator", self.report['statistical similarity']['avg_wd'],
        #         step=step
        #     )
        #     tf.summary.scalar(
        #         f"correlation_diff_aggregator",
        #         self.report['statistical similarity']['correlation_diff'], step=step
        #     )

            # tf.summary.scalar(f"metrics_averaged", metrics_averaged, step=step)
            # writer_federated.flush()

        return metrics_averaged, {}


config_path = sys.argv[-1]
# config_path = '../exp/adult/configs/config-ae-cat-fl.toml'
raw_config = load_config(config_path)
ds_name = raw_config['ds_name']
real_data_path = raw_config['real_data_path']
parent_dir = raw_config['parent_dir']
device = raw_config['device']

save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), config_path)

num_clients = raw_config['fl_params']['num_clients']
num_rounds = raw_config['fl_params']['num_rounds']
pretraining_rounds = raw_config['fl_params']['pretraining_rounds']

sys.stdout = Logger(parent_dir=raw_config['parent_dir'],
                    ds_name=raw_config['ds_name'],
                    encoding_type=raw_config['encoding_type'],
                    decoding_type=raw_config['decoding_type'],
                    stream=sys.stdout)
# sys.stdout.fileno = lambda: False

sys.stderr = Logger(parent_dir=raw_config['parent_dir'],
                    ds_name=raw_config['ds_name'],
                    encoding_type=raw_config['encoding_type'],
                    decoding_type=raw_config['decoding_type'],
                    stream=sys.stderr)
# sys.stderr.fileno = lambda: False

data, preprocessors, processed_dataset_info, _ = \
    get_dataset(raw_config['real_data_path'], raw_config['ds_info'], raw_config['parent_dir'],
                encoding=raw_config['encoding_type'], decoding=raw_config['decoding_type'],
                is_y_cond=raw_config['model_params']['is_y_cond'], auto_search=False, return_encoder=False,
                seed=raw_config['seed'], device=device)

X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test_tensor'], data['y_test_tensor']
train_length = len(X_train)
eval_length = len(X_val)
test_length = len(X_test)

dataloaders_dict = {}
batch_size = raw_config['train']['main']['batch_size']
if raw_config['ds_split'] == 'average' or raw_config['ds_split'] == 'uneven':
    split_type = raw_config['ds_split']
    train_split_dict = get_data_splits(train_length, num_clients, type=split_type)
    eval_split_dict = get_data_splits(eval_length, num_clients, type=split_type)
    test_split_dict = get_data_splits(test_length, num_clients, type=split_type)
    print(train_split_dict, eval_split_dict, test_split_dict)

    for i in range(num_clients):
        cid = str(i)
        X_train_split = X_train[train_split_dict[cid][0]:train_split_dict[cid][1]]
        y_train_split = y_train[train_split_dict[cid][0]:train_split_dict[cid][1]]
        train_dataset = torch.utils.data.TensorDataset(X_train_split, y_train_split)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        X_val_split = X_val[eval_split_dict[cid][0]:eval_split_dict[cid][1]]
        y_val_split = y_val[eval_split_dict[cid][0]:eval_split_dict[cid][1]]
        eval_dataset = torch.utils.data.TensorDataset(X_val_split, y_val_split)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
        X_test_split = X_test[test_split_dict[cid][0]:test_split_dict[cid][1]]
        y_test_split = y_test[test_split_dict[cid][0]:test_split_dict[cid][1]]
        test_dataset = torch.utils.data.TensorDataset(X_test_split, y_test_split)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        dataloaders = {
            'train': train_loader,
            'eval': eval_loader,
            'test': test_loader
        }
        dataloaders_dict[cid] = dataloaders

elif raw_config['ds_split'] == 'dirichlet':
    alpha = raw_config['alpha']
    train_client_idcs = dirichlet_split_noniid(train_labels=y_train, n_clients=num_clients, alpha=alpha, seed=raw_config['seed'])
    eval_client_idcs = dirichlet_split_noniid(train_labels=y_val, n_clients=num_clients, alpha=alpha, seed=raw_config['seed'])
    test_client_idcs = dirichlet_split_noniid(train_labels=y_test, n_clients=num_clients, alpha=alpha, seed=raw_config['seed'])

    for i in range(num_clients):
        cid = str(i)
        print(cid, len(train_client_idcs[i]))
        X_train_split = X_train[train_client_idcs[i]]
        y_train_split = y_train[train_client_idcs[i]]
        train_dataset = torch.utils.data.TensorDataset(X_train_split, y_train_split)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        X_val_split = X_val[eval_client_idcs[i]]
        y_val_split = y_val[eval_client_idcs[i]]
        eval_dataset = torch.utils.data.TensorDataset(X_val_split, y_val_split)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
        X_test_split = X_test[test_client_idcs[i]]
        y_test_split = y_test[test_client_idcs[i]]
        test_dataset = torch.utils.data.TensorDataset(X_test_split, y_test_split)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        dataloaders = {
            'train': train_loader,
            'eval': eval_loader,
            'test': test_loader
        }
        dataloaders_dict[cid] = dataloaders

# os.makedirs(parent_dir, exist_ok=True)
# encoder_clients = []
# diffusion_clients = []
# for i in range(num_clients):
#     client = EncoderClient(f'--config {config_path} --train --cid {i}', processed_dataset_info, dataloaders_dict[str(i)])
#     encoder_clients.append(client)
#
# # Create FedAvg strategy
# strategy = PretrainingSaveModelStrategy(
#         fraction_fit=1.0,  # Sample % of available clients for training
#         fraction_evaluate=1.0,  # Sample % of available clients for evaluation
#         min_fit_clients=2,  # Never sample less than n clients for training
#         min_evaluate_clients=2,  # Never sample less than n clients for evaluation
#         min_available_clients=num_clients,  # Wait until all n clients are available
#         on_fit_config_fn=fit_config,
#         on_evaluate_config_fn=fit_config
# )
#
# fl.simulation.start_simulation(
#     client_fn=lambda x: encoder_clients[int(x)],
#     num_clients=num_clients,
#     config=fl.server.ServerConfig(num_rounds=pretraining_rounds),
#     strategy=strategy,
#     # client_resources={"num_gpus": single_client_gpu_access, "num_cpus": single_client_cpu_access}
#     # ray_init_args={
#     #     "ignore_reinit_error": True,
#     #     "include_dashboard": False,
#     #     "num_gpus": 1
#     # },
# )


encoder_client = EncoderClient(f'--config {config_path} --train --cid {0}', processed_dataset_info, dataloaders_dict[str(0)])
aggregated_entity_encoder = encoder_client.entity_encoder

if os.path.exists(os.path.join(parent_dir, "ae_weights.npz")):
    parameters = np.load(os.path.join(parent_dir, "ae_weights.npz"))
    params_dict = zip(aggregated_entity_encoder.wrapper.model.state_dict().keys(), parameters.values())
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    aggregated_entity_encoder.wrapper.model.load_state_dict(state_dict, strict=True)
else:
    raise "Cannot find pretrained encoder weights."

if raw_config['encoding_type'] == 'ae_cat' and raw_config['decoding_type'] == '1nn':
    with open(os.path.join(parent_dir, '1nn_weights.pickle'), 'rb') as f:
        nn_classifiers = pickle.load(f)
    aggregated_entity_encoder.wrapper.nn_classifiers = nn_classifiers
print(f'log: Autoencoder weights loaded')

diffusion_clients = []
for i in range(num_clients):
    client = DiffusionClient(f'--config {config_path} --train --cid {i}', processed_dataset_info, preprocessors, dataloaders_dict[str(i)], aggregated_entity_encoder)
    diffusion_clients.append(client)

# Create FedAvg strategy
strategy = SaveModelStrategy(
        fraction_fit=1.0,  # Sample % of available clients for training
        fraction_evaluate=1.0,  # Sample % of available clients for evaluation
        min_fit_clients=2,  # Never sample less than n clients for training
        min_evaluate_clients=2,  # Never sample less than n clients for evaluation
        min_available_clients=num_clients,  # Wait until all n clients are available
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config
)
strategy.set_report_history(is_regression=processed_dataset_info['is_regression'])

# # Ray simulated
# fl.simulation.start_simulation(
#     client_fn=lambda x: diffusion_clients[int(x)],
#     num_clients=num_clients,
#     config=fl.server.ServerConfig(num_rounds=num_rounds),
#     strategy=strategy,
#     # client_resources={"num_gpus": single_client_gpu_access, "num_cpus": single_client_cpu_access}
#     client_resources={"num_gpus": 1},
#     ray_init_args={
#         "ignore_reinit_error": True,
#         "include_dashboard": False,
#         "num_gpus": 1
#     },
# )

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy
)

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=diffusion_clients[int(i)]
)

strategy.report_history.to_csv(os.path.join(parent_dir, 'aggregator_report_his.csv'), index=False)


if processed_dataset_info['is_regression']:
    report_history = pd.DataFrame(
        columns=['round', 'client', 'num_train', 'r2', 'rmse', 'mape', 'evs', 'disc f1', 'avg_jsd', 'avg_wd', 'correlation_diff'])
else:
    report_history = pd.DataFrame(
        columns=['round', 'client', 'num_train', 'acc', 'f1', 'roc_auc', 'disc f1', 'avg_jsd', 'avg_wd', 'correlation_diff'])

file_list = os.listdir(raw_config['parent_dir'])
model_paths_dict = {}
for i in file_list:
    if i.endswith('.pt') and 'clt' in i:
        client_id = i.split('-')[2]
        if client_id not in model_paths_dict:
            model_paths_dict[client_id] = [i]
        else:
            model_paths_dict[client_id].append(i)

for j in range(num_clients):
    model_paths = model_paths_dict[str(j)]
    model_paths = sorted(model_paths, key=lambda x: int(x.split('-')[-1].split('.')[0]))
    print('hola')
    print(model_paths)
    for i, model_path in enumerate(model_paths):
        model_path = os.path.join(raw_config['parent_dir'], model_path)
        if not os.path.exists(model_path):
            report_history.loc[len(report_history)] = [0] * len(report_history.columns)
        else:
            client_data = {'X_train': dataloaders_dict[str(j)]['train'].dataset.tensors[0],
                           'y_train': dataloaders_dict[str(j)]['train'].dataset.tensors[1],
                           'X_val': dataloaders_dict[str(j)]['eval'].dataset.tensors[0],
                           'y_val': dataloaders_dict[str(j)]['eval'].dataset.tensors[1],
                           'X_test': tensor2ndarray(dataloaders_dict[str(j)]['test'].dataset.tensors[0]),
                           'y_test': tensor2ndarray(dataloaders_dict[str(j)]['test'].dataset.tensors[1])}

            synthetic_data = sample(
                client_data,
                preprocessors,
                processed_dataset_info,
                aggregated_entity_encoder,
                num_samples=raw_config['sample']['num_samples'],
                batch_size=raw_config['sample']['batch_size'],
                disbalance=raw_config['sample'].get('disbalance', None),
                **raw_config['diffusion_params'],
                parent_dir=raw_config['parent_dir'],
                model_path=model_path,
                model_type=raw_config['model_type'],
                model_params=raw_config['model_params'],
                device=device,
                seed=raw_config['sample'].get('seed', 0),
                # change_val=args.change_val
            )

            report = train_catboost(
                raw_data=client_data,
                dataset_info=processed_dataset_info,
                ds_name=raw_config['ds_name'],
                parent_dir=raw_config['parent_dir'],
                preprocessors=preprocessors,
                seed=raw_config['seed'],
                eval_type=raw_config['eval']['type']['eval_type'],
                synthetic_data=synthetic_data,
                get_discriminator_measure=True,
                get_statistical_similarity=True
            )

            if processed_dataset_info['is_regression']:
                report_history.loc[len(report_history)] = [i + 1,
                                                           j,
                                                           len(dataloaders_dict[str(j)]['train'].dataset),
                                                           report['ml efficiency']['test']['r2'],
                                                           report['ml efficiency']['test']['rmse'],
                                                           report['ml efficiency']['test']['mape'],
                                                           report['ml efficiency']['test']['evs'],
                                                           report['discriminator measure']['test'][
                                                               'macro f1'],
                                                           report['statistical similarity']['avg_jsd'],
                                                           report['statistical similarity']['avg_wd'],
                                                           report['statistical similarity'][
                                                               'correlation_diff']]
            else:
                report_history.loc[len(report_history)] = [i + 1,
                                                           j,
                                                           len(dataloaders_dict[str(j)]['train'].dataset),
                                                           report['ml efficiency']['test']['acc'],
                                                           report['ml efficiency']['test']['f1'],
                                                           report['ml efficiency']['test']['roc_auc'],
                                                           report['discriminator measure']['test'][
                                                               'macro f1'],
                                                           report['statistical similarity']['avg_jsd'],
                                                           report['statistical similarity']['avg_wd'],
                                                           report['statistical similarity'][
                                                               'correlation_diff']]

report_history.to_csv(os.path.join(parent_dir, 'clients_report_his.csv'), index=False)



# # Serial training
# for i in range(num_clients):
#     cid = str(i)
#     client = diffusion_clients[i]
#     client.trainer.run_loop(client.encoder)
#     torch.save(client.diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, f'model-{cid}.pt'))
#     torch.save(client.trainer.ema_model.state_dict(), os.path.join(parent_dir, f'model_ema-{cid}.pt'))

# {'0': (0, 4607), '1': (4607, 9497), '2': (9497, 16670)} {'0': (0, 1887), '1': (1887, 3313), '2': (3313, 4168)} {'0': (0, 2301), '1': (2301, 2848), '2': (2848, 5210)}
# train_split_dict = {'0': (0, 4607), '1': (4607, 9497), '2': (9497, 16670)}
# total_samples = train_split_dict[str(num_clients-1)][1]
# aggregated_state_dict = {}
# unnormalized_weights = []
# for i in range(num_clients):
#     cid = str(i)
#     num_samples = train_split_dict[cid][1] - train_split_dict[cid][0]
#     weight = num_samples**2
#     unnormalized_weights.append(weight)
#     model_path = os.path.join(parent_dir, f'model-{cid}.pt')
#     client = diffusion_clients[i]
#     client.diffusion._denoise_fn.load_state_dict(
#         torch.load(model_path, map_location=device)
#     )
#     for key in client.diffusion._denoise_fn.state_dict():
#         if key not in aggregated_state_dict:
#             aggregated_state_dict[key] = [client.diffusion._denoise_fn.state_dict()[key]]
#         else:
#             aggregated_state_dict[key].append(client.diffusion._denoise_fn.state_dict()[key])
# unnormalized_weights = torch.Tensor(unnormalized_weights)
# weights = unnormalized_weights / torch.sum(unnormalized_weights)
# print(weights)
#
# for key in aggregated_state_dict:
#     values = aggregated_state_dict[key]
#     for i in range(len(values)):
#         values[i] *= weights[i]
#     values = torch.stack(values, dim=0)
#     aggregated_state_dict[key] = torch.mean(values, dim=0)
# torch.save(aggregated_state_dict, os.path.join(parent_dir, f'model-aggregation.pt'))
