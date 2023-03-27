import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import zero
import lib
import torch
import tomli
import shutil
import argparse
from train import train
from sample import sample
from eval_catboost import train_catboost
from eval_mlp import train_mlp
from eval_simple import train_simple
from scripts.eval_discriminator import train_discriminator
from entityencoder_utils.data_utils import get_dataset


def load_config(path) :
    with open(path, 'rb') as f:
        return tomli.load(f)
    
def save_file(parent_dir, config_path):
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--sample', action='store_true',  default=False)
    parser.add_argument('--eval', action='store_true',  default=False)
    parser.add_argument('--change_val', action='store_true',  default=False)

    args = parser.parse_args()
    raw_config = lib.load_config(args.config)
    if 'device' in raw_config:
        device = torch.device(raw_config['device'])
    else:
        device = torch.device('cuda:0')
    
    timer = zero.Timer()
    timer.run()
    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), args.config)

    sys.stdout = lib.Logger(parent_dir=raw_config['parent_dir'],
                            ds_name=raw_config['ds_name'],
                            encoding_type=raw_config['encoding_type'],
                            decoding_type=raw_config['decoding_type'],
                            stream=sys.stdout)
    sys.stderr = lib.Logger(parent_dir=raw_config['parent_dir'],
                            ds_name=raw_config['ds_name'],
                            encoding_type=raw_config['encoding_type'],
                            decoding_type=raw_config['decoding_type'],
                            stream=sys.stderr)

    encoder_params = raw_config['encoder']['encoder_params']
    encoder_batch_size = raw_config['encoder']['encoder_bs']

    data, preprocessors, processed_dataset_info, entity_encoder = \
        get_dataset(raw_config['real_data_path'], raw_config['ds_info'], raw_config['parent_dir'],
                    encoding=raw_config['encoding_type'], decoding=raw_config['decoding_type'],
                    is_y_cond=raw_config['model_params']['is_y_cond'], batch_size=encoder_batch_size,
                    auto_search=False, return_encoder=True, best_params=encoder_params,
                    seed=raw_config['seed'], device=device)

    if args.train:
        train(
            data,
            processed_dataset_info,
            entity_encoder,
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            seed=raw_config['seed'],
            device=device,
            # change_val=args.change_val
        )
    synthetic_data = None
    if args.sample:
        synthetic_data = sample(
            data,
            preprocessors,
            processed_dataset_info,
            entity_encoder,
            num_samples=raw_config['sample']['num_samples'],
            batch_size=raw_config['sample']['batch_size'],
            disbalance=raw_config['sample'].get('disbalance', None),
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            model_path=os.path.join(raw_config['parent_dir'], 'model.pt'),
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            device=device,
            seed=raw_config['sample'].get('seed', 0),
            # change_val=args.change_val
        )

    if args.eval:
        if raw_config['eval']['type']['eval_model'] == 'catboost':
            train_catboost(
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

            train_catboost(
                raw_data=data,
                dataset_info=processed_dataset_info,
                ds_name=raw_config['ds_name'],
                parent_dir=raw_config['parent_dir'],
                preprocessors=preprocessors,
                seed=raw_config['seed'],
                eval_type='real',
                synthetic_data=synthetic_data,
                get_discriminator_measure=False,
                get_statistical_similarity=False
            )

        # elif raw_config['eval']['type']['eval_model'] == 'mlp':
        #     train_mlp(
        #         parent_dir=raw_config['parent_dir'],
        #         real_data_path=raw_config['real_data_path'],
        #         eval_type=raw_config['eval']['type']['eval_type'],
        #         T_dict=raw_config['eval']['T'],
        #         seed=raw_config['seed'],
        #         change_val=args.change_val,
        #         device=device
        #     )
        # elif raw_config['eval']['type']['eval_model'] == 'simple':
        #     train_simple(
        #         parent_dir=raw_config['parent_dir'],
        #         real_data_path=raw_config['real_data_path'],
        #         eval_type=raw_config['eval']['type']['eval_type'],
        #         T_dict=raw_config['eval']['T'],
        #         seed=raw_config['seed'],
        #         change_val=args.change_val
        #     )

    print(f'Elapsed time: {str(timer)}')

if __name__ == '__main__':
    main()