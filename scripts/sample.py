from collections import OrderedDict

import torch
import numpy as np
import zero
import os

from ddpm import MLPDiffusion, GaussianDiffusion
from lib.util import tensor2ndarray
from lib import round_columns
import lib

def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)

def sample(
    data,
    preprocessers,
    processed_dataset_info,
    entity_encoder,
    parent_dir,
    batch_size = 2000,
    num_samples = 0,
    model_type = 'mlp',
    model_params = None,
    model_path = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    # T_dict = None,
    # num_numerical_features = 0,
    disbalance = None,
    device = torch.device('cuda:1'),
    seed = 0,
    change_val = False
):
    zero.improve_reproducibility(seed)

    # T = lib.Transformations(**T_dict)
    # D = make_dataset(
    #     real_data_path,
    #     T,
    #     num_classes=model_params['num_classes'],
    #     is_y_cond=model_params['is_y_cond'],
    #     change_val=change_val
    # )

    # K = np.array(D.get_category_sizes('train'))
    # if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
    #     K = np.array([0])

    n_cat = processed_dataset_info['n_cat']
    n_cat_emb = processed_dataset_info['n_cat_emb']
    n_num = processed_dataset_info['n_num']
    n_num_ = n_num + int(processed_dataset_info['is_regression'] and not model_params['is_y_cond'])
    d_in =  + n_cat_emb + n_num_
    model_params['d_in'] = int(d_in)

    model = MLPDiffusion(**model_params)
    if model_path.endswith('.pt'):
        model.load_state_dict(
            torch.load(model_path, map_location=device)
        )
    elif model_path.endswith('.npz'):
        parameters = np.load(model_path)
        params_dict = zip(model.state_dict().keys(), parameters.values())
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
    else:
        raise "Please provide model weights."
    print(model)

    embedding_sizes = processed_dataset_info['embedding_sizes']
    diffusion = GaussianDiffusion(
        embedding_sizes=embedding_sizes,
        num_features=model_params['d_in'],
        denoise_fn=model,
        loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        beta_scheduler=scheduler,
        device=device
    )
    diffusion.to(device)
    diffusion.eval()
    
    # _, empirical_class_dist = torch.unique(torch.from_numpy(D.y['train']), return_counts=True)
    # empirical_class_dist = empirical_class_dist.float() + torch.tensor([-5000., 10000.]).float()

    y_train = data['y_train']
    _, empirical_class_dist = torch.unique(y_train, return_counts=True)
    print(f'log: empirical_class_dist: {empirical_class_dist}')

    if disbalance == 'fix':
        empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], empirical_class_dist[0]
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)

    elif disbalance == 'fill':
        ix_major = empirical_class_dist.argmax().item()
        val_major = empirical_class_dist[ix_major].item()
        x_gen, y_gen = [], []
        for i in range(empirical_class_dist.shape[0]):
            if i == ix_major:
                continue
            distrib = torch.zeros_like(empirical_class_dist)
            distrib[i] = 1
            num_samples = val_major - empirical_class_dist[i].item()
            x_temp, y_temp = diffusion.sample_all(num_samples, batch_size, distrib.float(), ddim=False)
            x_gen.append(x_temp)
            y_gen.append(y_temp)
        
        x_gen = torch.cat(x_gen, dim=0)
        y_gen = torch.cat(y_gen, dim=0)

    else:
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)

    x_gen = entity_encoder.decode(x_gen.float().to(device))
    X_gen, y_gen = tensor2ndarray(x_gen), tensor2ndarray(y_gen)

    # # try:
    # # except FoundNANsError as ex:
    # #     print("Found NaNs during sampling!")
    # #     loader = lib.prepare_fast_dataloader(D, 'train', 8)
    # #     x_gen = next(loader)[0]
    # #     y_gen = torch.multinomial(
    # #         empirical_class_dist.float(),
    # #         num_samples=8,
    # #         replacement=True
    # #     )
    # X_gen, y_gen = x_gen.numpy(), y_gen.numpy()
    #
    # ###
    # # X_num_unnorm = X_gen[:, :num_numerical_features]
    # # lo = np.percentile(X_num_unnorm, 2.5, axis=0)
    # # hi = np.percentile(X_num_unnorm, 97.5, axis=0)
    # # idx = (lo < X_num_unnorm) & (hi > X_num_unnorm)
    # # X_gen = X_gen[np.all(idx, axis=1)]
    # # y_gen = y_gen[np.all(idx, axis=1)]
    # ###

    if n_num_ < X_gen.shape[1]:
        # np.save(os.path.join(parent_dir, 'X_cat_unnorm'), X_gen[:, num_numerical_features:])
        # # _, _, cat_encoder = lib.cat_encode({'train': X_cat_real}, T_dict['cat_encoding'], y_real, T_dict['seed'], True)
        # if T_dict['cat_encoding'] == 'one-hot':
        #     X_gen[:, num_numerical_features:] = to_good_ohe(D.cat_transform.steps[0][1], X_num_[:, num_numerical_features:])
        # X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])

        X_cat = tensor2ndarray(X_gen[:, :n_cat])
        print('log: generated categorical part: ', X_cat.shape)

    if n_num_ != 0:
        # _, normalize = lib.normalize({'train' : X_num_real}, T_dict['normalization'], T_dict['seed'], True)
        # np.save(os.path.join(parent_dir, 'X_num_unnorm'), X_gen[:, :num_numerical_features])

        normalizer = preprocessers['normalizer']
        X_num_ = normalizer.inverse_transform(X_gen[:, n_cat:])

        # # X_num_real = np.load(os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True)
        # X_num_real = tensor2ndarray(X_train)[:, n_cat:]
        # disc_cols = []
        # for col in range(X_num_real.shape[1]):
        #     uniq_vals = np.unique(X_num_real[:, col])
        #     if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
        #         disc_cols.append(col)
        # print("Discrete cols:", disc_cols)

        if n_num < n_num_:
            y_gen = X_num_[:, -1]
            X_num = X_num_[:, :-1]
        else:
            X_num = X_num_

        # if len(disc_cols):
        #     X_num = X_num_[:, :-1]
        #     X_num = round_columns(X_num_real, X_num, disc_cols)

    if n_num_ == 0:
        X_gen = X_cat
    elif n_cat == 0:
        X_gen = X_num
    else:
        X_gen = np.hstack([X_cat, X_num])

    np.save(os.path.join(parent_dir, 'X_train'), X_gen)
    print('log: synthetic dataset saved.')
    np.save(os.path.join(parent_dir, 'y_train'), y_gen)
    print('log: synthetic labels saved.')

    return X_gen, y_gen