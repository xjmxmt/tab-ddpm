import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import tomli
import shutil
import argparse
import torch
import torch.utils.data
from catboost import CatBoostError
import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict, Counter
import zero
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
from ddpm import GaussianDiffusion
from ddpm import MLPDiffusion
from scripts.train import Trainer


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


def load_data(cid: str, data_dict: dict, num_clients: int, batch_size: int, alpha=1.0, seed=42):
    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_val, y_val = data_dict['X_val'], data_dict['y_val']
    X_test, y_test = data_dict['X_test'], data_dict['y_test']

    train_client_idcs = dirichlet_split_noniid(train_labels=y_train, n_clients=num_clients, alpha=alpha, seed=seed)
    eval_client_idcs = dirichlet_split_noniid(train_labels=y_val, n_clients=num_clients, alpha=alpha, seed=seed)
    test_client_idcs = dirichlet_split_noniid(train_labels=y_test, n_clients=num_clients, alpha=alpha, seed=seed)

    cid = int(cid)
    X_train_split = X_train[train_client_idcs[cid]]
    y_train_split = y_train[train_client_idcs[cid]]

    print(f'Client-{cid}', f' Counter: {Counter(y_train_split.numpy().reshape(-1, ).tolist())}')
    # print(Counter(y_train.numpy().reshape(-1, ).tolist()))

    train_dataset = torch.utils.data.TensorDataset(X_train_split, y_train_split)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_val_split = X_val[eval_client_idcs[cid]]
    y_val_split = y_val[eval_client_idcs[cid]]
    eval_dataset = torch.utils.data.TensorDataset(X_val_split, y_val_split)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    X_test_split = X_test[test_client_idcs[cid]]
    y_test_split = y_test[test_client_idcs[cid]]

    dataloaders = {
        'train': train_loader,
        'val': eval_loader,
        'X_test': X_test_split,
        'y_test': y_test_split
    }
    return dataloaders


class DiffusionClient():

    def __init__(self, raw_config, processed_dataset_info, preprocessors, dataloaders, encoder, cid):

        self.raw_config = raw_config
        self.preprocessors = preprocessors
        self.processed_dataset_info = processed_dataset_info
        if processed_dataset_info['is_regression']:
            self.metrics_type = 'r2'
        else:
            self.metrics_type = 'f1'

        if 'device' in raw_config:
            device = torch.device(raw_config['device'])
        else:
            device = torch.device('cuda:0')

        seed = raw_config['seed']
        zero.improve_reproducibility(seed)

        self.device = device
        self.cid = cid
        self.parent_dir = raw_config['parent_dir']
        self.dataloaders = dataloaders
        self.encoder = encoder

        model_params = raw_config['model_params']
        model_params['device'] = device

        d_in = processed_dataset_info['n_cat_emb'] + processed_dataset_info['n_num'] + \
               int(processed_dataset_info['is_regression'] and not model_params['is_y_cond'])
        model_params['d_in'] = d_in
        print('log: d_in: ', model_params['d_in'])
        print(model_params)

        self.model = MLPDiffusion(**model_params)
        self.model.to(device)

        embedding_sizes = processed_dataset_info['embedding_sizes']
        gaussian_loss_type = raw_config['diffusion_params']['gaussian_loss_type']
        num_timesteps = raw_config['diffusion_params']['num_timesteps']
        scheduler = 'cosine'
        lr = raw_config['train']['main']['lr']
        weight_decay = raw_config['train']['main']['weight_decay']
        epochs = raw_config['train']['main']['epochs']

        self.diffusion = GaussianDiffusion(
            embedding_sizes=embedding_sizes,
            num_features=model_params['d_in'],
            denoise_fn=self.model,
            loss_type=gaussian_loss_type,
            num_timesteps=num_timesteps,
            beta_scheduler=scheduler,
            device=device,
            cid=self.cid
        )
        self.diffusion.to(device)
        self.diffusion.train()

        self.trainer = Trainer(
            self.diffusion,
            self.dataloaders['train'],
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            device=device
        )

    def fit(self, round):
        loss = self.trainer.run_loop(self.encoder)
        return loss


def aggregate(net_list, num_examples_list, weights=None):
    net_param_list = []
    for net in net_list:
        ndarray = [val.cpu().numpy() for _, val in net.state_dict().items()]
        net = fl.common.ndarrays_to_parameters(ndarray)
        net_param_list.append(net)

    if weights is None:
        weights_results = [
            (fl.common.parameters_to_ndarrays(parameters), num_examples)
            for parameters, num_examples in zip(net_param_list, num_examples_list)
        ]
    else:
        weights_results = [
            (fl.common.parameters_to_ndarrays(parameters), num_examples*weight)
            for parameters, num_examples, weight in zip(net_param_list, num_examples_list, weights)
        ]
    aggregated_parameters = fl.server.strategy.aggregate.aggregate(weights_results)
    return aggregated_parameters


def update_report_history(round, report, report_history, is_regression):
    if is_regression:
        report_history.loc[len(report_history)] = [round,
                                                   report['ml efficiency']['test']['r2'],
                                                   report['ml efficiency']['test']['rmse'],
                                                   report['ml efficiency']['test']['mape'],
                                                   report['ml efficiency']['test']['evs'],
                                                   report['discriminator measure']['test']['macro f1'],
                                                   report['statistical similarity']['avg_jsd'],
                                                   report['statistical similarity']['avg_wd'],
                                                   report['statistical similarity']['correlation_diff']]
    else:
        report_history.loc[len(report_history)] = [round,
                                                   report['ml efficiency']['test']['acc'],
                                                   report['ml efficiency']['test']['f1'],
                                                   report['ml efficiency']['test']['roc_auc'],
                                                   report['discriminator measure']['test']['macro f1'],
                                                   report['statistical similarity']['avg_jsd'],
                                                   report['statistical similarity']['avg_wd'],
                                                   report['statistical similarity']['correlation_diff']]
    return report_history


parser = argparse.ArgumentParser()
parser.add_argument('--config', metavar='FILE')
args = parser.parse_args()

# config_path = sys.argv[-1]
config_path = args.config
# config_path = '../exp/adult/configs/config-ae-cat-fl.toml'
print('config path: ', config_path)

raw_config = load_config(config_path)
save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), config_path)

sys.stdout = Logger(parent_dir=raw_config['parent_dir'],
                    ds_name=raw_config['ds_name'],
                    encoding_type=raw_config['encoding_type'],
                    decoding_type=raw_config['decoding_type'],
                    stream=sys.stdout)

sys.stderr = Logger(parent_dir=raw_config['parent_dir'],
                    ds_name=raw_config['ds_name'],
                    encoding_type=raw_config['encoding_type'],
                    decoding_type=raw_config['decoding_type'],
                    stream=sys.stderr)

parent_dir = raw_config['parent_dir']
seed = raw_config['seed']
device = raw_config['device']
num_clients = raw_config['fl_params']['num_clients']
pretraining_rounds = raw_config['fl_params']['pretraining_rounds']
num_rounds = raw_config['fl_params']['num_rounds']

data, preprocessors, processed_dataset_info, _ = \
    get_dataset(raw_config['real_data_path'], raw_config['ds_info'], raw_config['parent_dir'],
                encoding=raw_config['encoding_type'], decoding=raw_config['decoding_type'],
                is_y_cond=raw_config['model_params']['is_y_cond'], auto_search=False, return_encoder=False,
                seed=raw_config['seed'], device=device)

X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

data = {'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val, 'X_test': X_test, 'y_test': y_test}

encoder_params = raw_config['encoder']['encoder_params']
epochs = encoder_params['epochs']
lr = encoder_params['lr']
batch_size = raw_config['encoder']['encoder_bs']
alpha = raw_config['alpha']

model_params = {}
if 'latent_dim' in encoder_params:
    model_params['latent_dim'] = encoder_params['latent_dim']
if 'n_layers' in encoder_params:
    model_params['n_layers'] = encoder_params['n_layers']
if 'using_noise' in encoder_params:
    model_params['using_noise'] = encoder_params['using_noise']
if 'emb_activation' in encoder_params:
    model_params['emb_activation'] = encoder_params['emb_activation']
if 'cat_ratio' in encoder_params:
    model_params['cat_ratio'] = encoder_params['cat_ratio']

"""
 training entity encoders
"""
round = 1
encoder_dict = {}
encoder_model_list = []
num_examples_dict = {}
dataloaders_dict = {}

for cid in range(num_clients):
    cid = str(cid)
    dataloaders = load_data(cid, data, num_clients, batch_size, alpha=alpha, seed=seed)
    num_examples_dict[cid] = len(dataloaders['train'].dataset)
    dataloaders_dict[cid] = dataloaders
    entity_encoder = Encoder(parent_dir, processed_dataset_info,
                             encoding_type=raw_config['encoding_type'], decoding_type=raw_config['decoding_type'],
                             encoding_y=processed_dataset_info['including_y'], params=model_params,
                             seed=seed, device=device)

    # Note: uncomment this when training
    entity_encoder.fit(None, None, save_path='auto_searching', epochs=epochs, batch_size=batch_size,
                       lr=lr, dataloader=dataloaders['train'])

    # # save weights
    # print(f'Saving weights in Round-{round} Client-{cid}.')
    # torch.save(entity_encoder.wrapper.model.state_dict(), os.path.join(parent_dir, f'encoder-clt-{cid}-{round}.pt'))

    encoder_dict[cid] = entity_encoder
    encoder_model_list.append(entity_encoder.wrapper.model)

aggregated_ndarrays = aggregate(encoder_model_list, num_examples_dict.values())
params_dict = zip(entity_encoder.wrapper.model.state_dict().keys(), aggregated_ndarrays)
state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

for round in range(2, pretraining_rounds+1):
    print(f'Round-{round} starts.')
    encoder_model_list = []

    for cid in range(num_clients):
        print(f'Client-{cid} starts.')
        cid = str(cid)
        entity_encoder = encoder_dict[cid]
        dataloaders = dataloaders_dict[cid]

        # load aggregated weights
        entity_encoder.wrapper.model.load_state_dict(state_dict, strict=True)

        # train locally
        entity_encoder.fit(None, None, save_path='auto_searching', epochs=epochs, batch_size=batch_size,
                           lr=lr, dataloader=dataloaders['train'])
        encoder_model_list.append(entity_encoder.wrapper.model)
        encoder_dict[cid] = entity_encoder

        # # save weights
        # print(f'Saving weights in Round-{round} Client-{cid}.')
        # torch.save(entity_encoder.wrapper.model.state_dict(), os.path.join(parent_dir, f'encoder-clt-{cid}-{round}.pt'))

    # save aggregated weights at the end of round
    aggregated_ndarrays = aggregate(encoder_model_list, num_examples_dict.values())
    params_dict = zip(entity_encoder.wrapper.model.state_dict().keys(), aggregated_ndarrays)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

# load the final aggregated encoder weights
aggregated_encoder = entity_encoder
aggregated_encoder.wrapper.model.load_state_dict(state_dict, strict=True)

# save final aggregated weights
print(f'Saving round {pretraining_rounds} aggregated ndarrays.')
np.savez(os.path.join(parent_dir, f"encoder-aggregated.npz"), *aggregated_ndarrays)


"""
 diffusion models training
"""
if processed_dataset_info['is_regression']:
    report_history = pd.DataFrame(
        columns=['round', 'r2', 'rmse', 'mape', 'evs', 'disc f1', 'avg_jsd', 'avg_wd', 'correlation_diff'])
else:
    report_history = pd.DataFrame(
        columns=['round', 'acc', 'f1', 'roc_auc', 'disc f1', 'avg_jsd', 'avg_wd', 'correlation_diff'])

round = 1
diffusion_dict = {}
diffusion_model_list = []
avg_loss_dict = {}
avg_loss_list = []

for cid in range(num_clients):
    cid = str(cid)

    # create diffusion client and start training
    dataloaders = dataloaders_dict[cid]
    diffusion_client = DiffusionClient(raw_config, processed_dataset_info, preprocessors, dataloaders, aggregated_encoder, cid)
    avg_loss = diffusion_client.fit(round)
    avg_loss_list.append(1/avg_loss)
    print(f'Saving weights in Client-{cid}.')
    torch.save(diffusion_client.diffusion._denoise_fn.state_dict(),
               os.path.join(parent_dir, f'model-clt-{cid}-{round}.pt'))

    # store diffusion model
    diffusion_dict[cid] = diffusion_client
    diffusion_model_list.append(diffusion_client.model)

# aggregate model weights
aggregated_ndarrays = aggregate(diffusion_model_list, num_examples_dict.values(), weights=avg_loss_list)
avg_loss_dict[round] = avg_loss_list
print('avgerage loss: ', avg_loss_list)
# aggregated_ndarrays = aggregate(diffusion_model_list, num_examples_dict.values())
params_dict = zip(diffusion_client.model.state_dict().keys(), aggregated_ndarrays)
state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

print(f'Saving round {round} aggregated ndarrays.')
np.savez(os.path.join(parent_dir, f"model-rnd-{round}.npz"), *aggregated_ndarrays)

# testing
synthetic_data = sample(
    data,
    preprocessors,
    processed_dataset_info,
    aggregated_encoder,
    num_samples=raw_config['sample']['num_samples'],
    batch_size=raw_config['sample']['batch_size'],
    disbalance=raw_config['sample'].get('disbalance', None),
    **raw_config['diffusion_params'],
    parent_dir=raw_config['parent_dir'],
    model_path=os.path.join(raw_config['parent_dir'], f"model-rnd-{round}.npz"),
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

report_history = update_report_history(round, report, report_history, processed_dataset_info['is_regression'])

for round in range(2, num_rounds+1):
    print(f'Round-{round} starts.')
    diffusion_model_list = []

    avg_loss_list = []

    for cid in range(num_clients):
        print(f'Client-{cid} starts.')
        cid = str(cid)
        dataloaders = dataloaders_dict[cid]
        diffusion_client = diffusion_dict[cid]

        diffusion_client.model.load_state_dict(state_dict, strict=True)
        avg_loss = diffusion_client.fit(round)
        avg_loss_list.append(1/avg_loss)

        print(f'Saving weights in Client-{cid}.')
        torch.save(diffusion_client.diffusion._denoise_fn.state_dict(),
                   os.path.join(parent_dir, f'model-clt-{cid}-{round}.pt'))

        # store diffusion model
        diffusion_dict[cid] = diffusion_client
        diffusion_model_list.append(diffusion_client.model)

    aggregated_ndarrays = aggregate(diffusion_model_list, num_examples_dict.values(), weights=avg_loss_list)
    avg_loss_dict[round] = avg_loss_list
    print('average loss: ', avg_loss_list)
    # aggregated_ndarrays = aggregate(diffusion_model_list, num_examples_dict.values())
    params_dict = zip(diffusion_client.model.state_dict().keys(), aggregated_ndarrays)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

    print(f'Saving round {round} aggregated ndarrays.')
    np.savez(os.path.join(parent_dir, f"model-rnd-{round}.npz"), *aggregated_ndarrays)

    synthetic_data = sample(
        data,
        preprocessors,
        processed_dataset_info,
        aggregated_encoder,
        num_samples=raw_config['sample']['num_samples'],
        batch_size=raw_config['sample']['batch_size'],
        disbalance=raw_config['sample'].get('disbalance', None),
        **raw_config['diffusion_params'],
        parent_dir=raw_config['parent_dir'],
        model_path=os.path.join(raw_config['parent_dir'], f"model-rnd-{round}.npz"),
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

    report_history = update_report_history(round, report, report_history, processed_dataset_info['is_regression'])

report_history.to_csv(os.path.join(parent_dir, 'aggregator_report_history.csv'), index=False)

"""
 test all intermediate models
"""

# Note:
#  comment this block when training

# parameters = np.load(os.path.join(parent_dir, f"encoder-aggregated.npz"))
# params_dict = zip(entity_encoder.wrapper.model.state_dict().keys(), parameters.values())
# state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
# entity_encoder.wrapper.model.load_state_dict(state_dict, strict=True)
# aggregated_encoder = entity_encoder


if processed_dataset_info['is_regression']:
    report_history = pd.DataFrame(
        columns=['round', 'client', 'num_train', 'reverse_avg_loss', 'r2', 'rmse', 'mape', 'evs', 'disc f1', 'avg_jsd', 'avg_wd', 'correlation_diff'])
else:
    report_history = pd.DataFrame(
        columns=['round', 'client', 'num_train', 'reverse_avg_loss', 'acc', 'f1', 'roc_auc', 'disc f1', 'avg_jsd', 'avg_wd', 'correlation_diff'])

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
    cid = str(j)
    dataloaders = dataloaders_dict[cid]
    model_paths = model_paths_dict[cid]
    model_paths = sorted(model_paths, key=lambda x: int(x.split('-')[-1].split('.')[0]))

    # extract data from dataloaders
    client_data = {'X_train': dataloaders['train'].dataset.tensors[0],
                   'y_train': dataloaders['train'].dataset.tensors[1],
                   'X_val': dataloaders['val'].dataset.tensors[0],
                   'y_val': dataloaders['val'].dataset.tensors[1],
                   'X_test': dataloaders['X_test'],
                   'y_test': dataloaders['y_test']}

    for i, model_path in enumerate(model_paths):
        model_path = os.path.join(raw_config['parent_dir'], model_path)
        if not os.path.exists(model_path):
            report_history.loc[len(report_history)] = [0] * len(report_history.columns)
        else:
            print('sampling with model: ', model_path)
            synthetic_data = sample(
                client_data,
                preprocessors,
                processed_dataset_info,
                aggregated_encoder,
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

            # try:
            #     report = train_catboost(
            #         raw_data=client_data,
            #         dataset_info=processed_dataset_info,
            #         ds_name=raw_config['ds_name'],
            #         parent_dir=raw_config['parent_dir'],
            #         preprocessors=preprocessors,
            #         seed=raw_config['seed'],
            #         eval_type=raw_config['eval']['type']['eval_type'],
            #         synthetic_data=synthetic_data,
            #         get_discriminator_measure=True,
            #         get_statistical_similarity=True
            #     )
            # except CatBoostError:
            #     continue

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
                                                           len(client_data['X_train']),
                                                           avg_loss_dict[i + 1][j],
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
                                                           len(client_data['X_train']),
                                                           avg_loss_dict[i + 1][j],
                                                           report['ml efficiency']['test']['acc'],
                                                           report['ml efficiency']['test']['f1'],
                                                           report['ml efficiency']['test']['roc_auc'],
                                                           report['discriminator measure']['test'][
                                                               'macro f1'],
                                                           report['statistical similarity']['avg_jsd'],
                                                           report['statistical similarity']['avg_wd'],
                                                           report['statistical similarity'][
                                                               'correlation_diff']]

report_history.to_csv(os.path.join(parent_dir, 'clients_report_history.csv'), index=False)
