import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from collections import OrderedDict
import torch
import numpy as np
import flwr as fl
import tomli
import pickle
import torch.nn as nn
import torch.utils.data
import argparse
import zero
from lib.logger import Logger
from entity_encoders.encoder import Encoder
from fl_simulation.util import dirichlet_split_noniid
from entityencoder_utils.data_utils import get_dataset


def load_config(path):
    with open(path, 'rb') as f:
        return tomli.load(f)


class EncoderClient(fl.client.NumPyClient):

    def __init__(self, cid: str, config_path: str, processed_dataset_info: dict, dataloaders):
        raw_config = load_config(config_path)

        # sys.stdout = Logger(parent_dir=raw_config['parent_dir'],
        #                     ds_name=raw_config['ds_name'],
        #                     encoding_type=raw_config['encoding_type'],
        #                     decoding_type=raw_config['decoding_type'],
        #                     comment="client-" + cid,
        #                     stream=sys.stdout)
        #
        # sys.stderr = Logger(parent_dir=raw_config['parent_dir'],
        #                     ds_name=raw_config['ds_name'],
        #                     encoding_type=raw_config['encoding_type'],
        #                     decoding_type=raw_config['decoding_type'],
        #                     comment="client-" + cid,
        #                     stream=sys.stderr)

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

        encoder_params = raw_config['train']['main']['encoder_params']
        self.epochs = raw_config['train']['main']['encoder_epochs']
        self.lr = raw_config['train']['main']['encoder_lr']
        self.batch_size = raw_config['train']['main']['encoder_bs']

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
        if 'epochs' in encoder_params:
            self.epochs = encoder_params['epochs']
        if 'lr' in encoder_params:
            self.lr = encoder_params['lr']

        self.entity_encoder = Encoder(raw_config['parent_dir'], processed_dataset_info,
                                      encoding_type=raw_config['encoding_type'], decoding_type=raw_config['decoding_type'],
                                      encoding_y=processed_dataset_info['including_y'], params=model_params,
                                      seed=raw_config['seed'], device=device)
        self.net = self.entity_encoder.wrapper.model

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.entity_encoder.fit(None, None, save_path=self.parent_dir, epochs=self.epochs, batch_size=self.batch_size,
                                lr=self.lr, dataloader=self.dataloaders['train'])
        return self.get_parameters(config), len(self.dataloaders['train'].dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = float('inf')
        return loss, len(self.dataloaders['eval'].dataset), {}


def load_data(cid, data_dict, num_clients, batch_size, alpha=1.0, seed=42):
    X_train, y_train = data_dict['X_train'], data_dict['y_train']
    X_test, y_test = data_dict['X_test'], data_dict['y_test']

    train_client_idcs = dirichlet_split_noniid(train_labels=y_train, n_clients=num_clients, alpha=alpha, seed=seed)
    test_client_idcs = dirichlet_split_noniid(train_labels=y_test, n_clients=num_clients, alpha=alpha, seed=seed)

    cid = int(cid)
    X_train_split = X_train[train_client_idcs[cid]]
    y_train_split = y_train[train_client_idcs[cid]]
    train_dataset = torch.utils.data.TensorDataset(X_train_split, y_train_split)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_test_split = X_test[test_client_idcs[cid]]
    y_test_split = y_test[test_client_idcs[cid]]
    test_dataset = torch.utils.data.TensorDataset(X_test_split, y_test_split)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    dataloaders = {
        'train': train_loader,
        'test': test_loader
    }
    return dataloaders


parser = argparse.ArgumentParser()
parser.add_argument('--config', metavar='FILE')
parser.add_argument('--cid', required=True)

args = parser.parse_args()
config_path = args.config
# config_path = '../exp/adult/configs/config-ae-cat-fl.toml'
print('config path: ', config_path)

raw_config = load_config(config_path)
ds_name = raw_config['ds_name']
real_data_path = raw_config['real_data_path']
parent_dir = raw_config['parent_dir']
device = raw_config['device']

data, preprocessors, processed_dataset_info, _ = \
    get_dataset(raw_config['real_data_path'], raw_config['ds_info'], raw_config['parent_dir'],
                encoding=raw_config['encoding_type'], decoding=raw_config['decoding_type'],
                is_y_cond=raw_config['model_params']['is_y_cond'], auto_search=False, return_encoder=False,
                seed=raw_config['seed'], device=device)

X_train, y_train = data['X_train'], data['y_train']
X_eval, y_eval = data['X_val'], data['y_val']
X_test, y_test = data['X_test_tensor'], data['y_test_tensor']

X_train = torch.vstack([X_train, X_eval])
y_train = torch.vstack([y_train.reshape(-1, 1), y_eval.reshape(-1, 1)])
new_data = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}

dataloaders = load_data(args.cid, new_data, raw_config['fl_params']['num_clients'],
                        raw_config['train']['main']['encoder_bs'], raw_config['alpha'], raw_config['seed'])

fl.client.start_numpy_client(
    server_address="127.0.0.1:9999",
    client=EncoderClient(args.cid, config_path, processed_dataset_info, dataloaders),
)