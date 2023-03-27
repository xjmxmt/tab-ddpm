import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)


from collections import OrderedDict
import torch
import flwr as fl
import time
import tomli
import torch.utils.data
import argparse
import zero
from scripts.train import Trainer
from lib.logger import Logger
from ddpm import GaussianDiffusion
from ddpm import MLPDiffusion
from scripts.sample import sample
from scripts.eval_catboost import train_catboost
from lib.util import tensor2ndarray


def load_config(path):
    with open(path, 'rb') as f:
        return tomli.load(f)


class DiffusionClient(fl.client.NumPyClient):

    def __init__(self, string, processed_dataset_info, preprocessors, dataloaders, encoder):
        """Deal with commend line arguments"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', metavar='FILE')
        parser.add_argument('--train', action='store_true', default=False)
        parser.add_argument('--sample', action='store_true', default=False)
        parser.add_argument('--eval', action='store_true', default=False)
        parser.add_argument('--cid', required=True)

        # Note: for simulation
        args = parser.parse_args(string.split())
        raw_config = load_config(args.config)

        self.raw_config = raw_config
        self.preprocessors = preprocessors
        self.processed_dataset_info = processed_dataset_info
        if processed_dataset_info['is_regression']:
            self.metrics_type = 'r2'
        else:
            self.metrics_type = 'f1'

        sys.stdout = Logger(parent_dir=raw_config['parent_dir'],
                            ds_name=raw_config['ds_name'],
                            encoding_type=raw_config['encoding_type'],
                            decoding_type=raw_config['decoding_type'],
                            comment="client-"+args.cid,
                            stream=sys.stdout)

        sys.stderr = Logger(parent_dir=raw_config['parent_dir'],
                            ds_name=raw_config['ds_name'],
                            encoding_type=raw_config['encoding_type'],
                            decoding_type=raw_config['decoding_type'],
                            comment="client-" + args.cid,
                            stream=sys.stderr)

        if 'device' in raw_config:
            device = torch.device(raw_config['device'])
        else:
            device = torch.device('cuda:0')

        seed = raw_config['seed']
        zero.improve_reproducibility(seed)

        self.device = device
        self.cid = args.cid
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
        self.net = self.model
        # print('log: model structure: ', self.model)

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

        self.current_round = 0

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

        self.diffusion._denoise_fn = self.net
        self.trainer.diffusion = self.diffusion

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loss = self.trainer.run_loop(self.encoder)

        print(f'Saving weights in Client-{self.cid}.')
        torch.save(self.diffusion._denoise_fn.state_dict(), os.path.join(self.parent_dir, f'model-clt-{self.cid}-{int(time.time())}.pt'))

        return self.get_parameters(config), len(self.dataloaders['train'].dataset), {'loss': loss}

    def evaluate(self, parameters, config):
        # self.set_parameters(parameters)

        # data = {'X_train': self.dataloaders['train'].dataset.tensors[0],
        #         'y_train': self.dataloaders['train'].dataset.tensors[1],
        #         'X_val': self.dataloaders['eval'].dataset.tensors[0],
        #         'y_val': self.dataloaders['eval'].dataset.tensors[1],
        #         'X_test': tensor2ndarray(self.dataloaders['test'].dataset.tensors[0]),
        #         'y_test': tensor2ndarray(self.dataloaders['test'].dataset.tensors[1])}
        #
        # synthetic_data = sample(
        #     data,
        #     self.preprocessors,
        #     self.processed_dataset_info,
        #     self.encoder,
        #     num_samples=self.raw_config['sample']['num_samples'],
        #     batch_size=self.raw_config['sample']['batch_size'],
        #     disbalance=self.raw_config['sample'].get('disbalance', None),
        #     **self.raw_config['diffusion_params'],
        #     parent_dir=self.raw_config['parent_dir'],
        #     model_path=os.path.join(self.raw_config['parent_dir'], f'model-{self.current_round}-{self.cid}.pt'),
        #     model_type=self.raw_config['model_type'],
        #     model_params=self.raw_config['model_params'],
        #     device=self.device,
        #     seed=self.raw_config['sample'].get('seed', 0),
        #     # change_val=args.change_val
        # )
        #
        # report = train_catboost(
        #     raw_data=data,
        #     dataset_info=self.processed_dataset_info,
        #     ds_name=self.raw_config['ds_name'],
        #     parent_dir=self.raw_config['parent_dir'],
        #     preprocessors=self.preprocessors,
        #     seed=self.raw_config['seed'],
        #     eval_type=self.raw_config['eval']['type']['eval_type'],
        #     synthetic_data=synthetic_data,
        #     get_discriminator_measure=True,
        #     get_statistical_similarity=True
        # )
        # return report['ml efficiency'][self.metrics_type], len(self.dataloaders['eval'].dataset), {'report': report}

        return 0.0, len(self.dataloaders['eval'].dataset), {}
