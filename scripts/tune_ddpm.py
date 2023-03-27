import subprocess
import lib
import os
import optuna
from copy import deepcopy
import tomli
import shutil
import argparse
from pathlib import Path


def load_config(path):
    with open(path, 'rb') as f:
        return tomli.load(f)


parser = argparse.ArgumentParser()
parser.add_argument('--config', metavar='FILE')
# parser.add_argument('ds_name', type=str)
# parser.add_argument('train_size', type=int)
# parser.add_argument('eval_type', type=str)
# parser.add_argument('eval_model', type=str)
# parser.add_argument('prefix', type=str)
# parser.add_argument('--eval_seeds', action='store_true',  default=False)

args = parser.parse_args()
config_path = args.config
raw_config = load_config(config_path)

ds_name = raw_config['ds_name']
eval_type = raw_config['eval']['type']['eval_type']
assert eval_type in ('merged', 'synthetic')
train_size = raw_config['train_size']
prefix = raw_config['eval']['type']['eval_model']

pipeline = f'scripts/pipeline.py'
base_config_path = config_path
parent_path = raw_config['parent_dir']
exps_path = Path(f'exp/{ds_name}/many-exps/') # temporary dir. maybe will be replaced with tempdiвdr
eval_seeds = f'scripts/eval_seeds.py'

os.makedirs(exps_path, exist_ok=True)

def _suggest_mlp_layers(trial):
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t
    min_n_layers, max_n_layers, d_min, d_max = 1, 4, 7, 10
    n_layers = 2 * trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    return d_layers

def objective(trial):
    
    lr = trial.suggest_loguniform('lr', 0.00001, 0.003)
    d_layers = _suggest_mlp_layers(trial)
    weight_decay = 0.0    
    batch_size = trial.suggest_categorical('batch_size', [256, 4096])
    epochs = trial.suggest_categorical('epochs', [500, 2000, 5000, 12000, 20000])
    # steps = trial.suggest_categorical('steps', [500]) # for debug
    gaussian_loss_type = 'mse'
    # scheduler = trial.suggest_categorical('scheduler', ['cosine', 'linear'])
    num_timesteps = trial.suggest_categorical('num_timesteps', [100, 1000])
    num_samples = int(train_size * (2 ** trial.suggest_int('num_samples', -2, 1)))

    # entity encoder params
    encoder_lr = trial.suggest_loguniform('encoder_lr', 0.00001, 0.001)
    encoder_epochs = trial.suggest_categorical('encoder_epochs', [500, 1000, 2000])
    latent_dim = trial.suggest_categorical('latent_dim', [None, 'half', 'one_third', 'sqrt'])
    encoder_n_layers = trial.suggest_categorical('encoder_n_layers', [2, 3, 5])

    base_config = lib.load_config(base_config_path)

    base_config['train']['main']['lr'] = lr
    base_config['train']['main']['epochs'] = epochs
    base_config['train']['main']['batch_size'] = batch_size
    base_config['train']['main']['weight_decay'] = weight_decay
    base_config['model_params']['rtdl_params']['d_layers'] = d_layers
    base_config['eval']['type']['eval_type'] = eval_type
    base_config['sample']['num_samples'] = num_samples
    base_config['diffusion_params']['gaussian_loss_type'] = gaussian_loss_type
    base_config['diffusion_params']['num_timesteps'] = num_timesteps
    # base_config['diffusion_params']['scheduler'] = scheduler

    base_config['parent_dir'] = str(exps_path / f"{trial.number}")
    base_config['eval']['type']['eval_model'] = raw_config['eval']['type']['eval_model']
    if raw_config['eval']['type']['eval_model'] == "mlp":
        base_config['eval']['T']['normalization'] = "quantile"
        base_config['eval']['T']['cat_encoding'] = "one-hot"

    base_config['encoder_params'] = {'lr': encoder_lr, 'epochs': encoder_epochs, 'n_layers': encoder_n_layers, 'latent_dim': latent_dim}

    trial.set_user_attr("config", base_config)

    lib.dump_config(base_config, exps_path / 'config.toml')

    subprocess.run(['python', f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--train'], check=True)

    n_datasets = 1
    score = 0.0

    for sample_seed in range(n_datasets):
        base_config['sample']['seed'] = sample_seed
        lib.dump_config(base_config, exps_path / 'config.toml')
        
        subprocess.run(['python', f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--sample', '--eval'], check=True)

        report_path = str(Path(base_config['parent_dir']) / f'results_{raw_config["eval"]["type"]["eval_model"]}.json')
        report = lib.load_json(report_path)

        if 'r2' in report['ml efficiency']['test']:
            score += report['ml efficiency']['test']['r2']
        else:
            score += report['ml efficiency']['test']['f1']

    shutil.rmtree(exps_path / f"{trial.number}")

    return score / n_datasets

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

best_config_path = parent_path / f'{prefix}_best/config.toml'
best_config = study.best_trial.user_attrs['config']
best_config["parent_dir"] = str(parent_path / f'{prefix}_best/')

os.makedirs(parent_path / f'{prefix}_best', exist_ok=True)
lib.dump_config(best_config, best_config_path)
lib.dump_json(optuna.importance.get_param_importances(study), parent_path / f'{prefix}_best/importance.json')

# subprocess.run(['python3.9', f'{pipeline}', '--config', f'{best_config_path}', '--train', '--sample'], check=True)

# if args.eval_seeds:
#     best_exp = str(parent_path / f'{prefix}_best/config.toml')
#     subprocess.run(['python3.9', f'{eval_seeds}', '--config', f'{best_exp}', '10', "ddpm", eval_type, args.eval_model, '5'], check=True)
