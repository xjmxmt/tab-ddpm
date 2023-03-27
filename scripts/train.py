from copy import deepcopy
import torch
import torch.utils.data
import os
import numpy as np
import zero
from ddpm import MLPDiffusion, GaussianDiffusion
from scripts.utils_train import update_ema
import pandas as pd

class Trainer:
    def __init__(self, diffusion, train_loader, lr, weight_decay, epochs, device=torch.device('cuda:0')):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_loader = train_loader
        self.epochs = epochs
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'gloss'])
        self.log_every = 10
        self.print_every = 100
        self.ema_every = 1000

    def _anneal_lr(self, epoch):
        frac_done = epoch / self.epochs
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, y_dict):
        x = x.to(self.device)
        for k in y_dict:
            y_dict[k] = y_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_gauss = self.diffusion.training_losses(self.diffusion._denoise_fn, x, y_dict)
        loss = loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_gauss

    def run_loop(self, encoder):
        curr_loss_gauss = 0.0
        curr_count = 0
        gloss = 0.0

        print('Training started.')
        for epoch in range(self.epochs):
            for idx, batch in enumerate(self.train_loader):
                batch_x, batch_y = batch
                batch_x = encoder.encode(batch_x)
                batch_x.to(self.device)
                batch_y.long().to(self.device)
                y_dict = {'y': batch_y}
                batch_loss_gauss = self._run_step(batch_x, y_dict)

                self._anneal_lr(epoch)

                curr_count += len(batch_x)
                curr_loss_gauss += batch_loss_gauss.item() * len(batch_x)

                update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            if (epoch + 1) % self.log_every == 0:
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (epoch + 1) % self.print_every == 0:
                    print(f'Client {self.diffusion.cid} Epoch {(epoch + 1)}/{self.epochs} GLoss: {gloss}')
                self.loss_history.loc[len(self.loss_history)] = [epoch + 1, gloss]
                curr_count = 0
                curr_loss_gauss = 0.0

        # return the average training loss
        avg_loss = np.mean(self.loss_history.values[:, -1], keepdims=False)
        if isinstance(avg_loss, np.ndarray):
            avg_loss = avg_loss[0]
        return avg_loss


def train(
    data,
    processed_dataset_info,
    entity_encoder,
    parent_dir,
    real_data_path = 'data/adult/train.csv',
    epochs = 1000,
    lr = 0.002,
    weight_decay = 1e-4,
    batch_size = 1024,
    model_type = 'mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    device = torch.device('cuda:1'),
    seed = 0,
    # change_val = False
):
    real_data_path = os.path.normpath(real_data_path)
    parent_dir = os.path.normpath(parent_dir)

    zero.improve_reproducibility(seed)

    d_in = processed_dataset_info['n_cat_emb'] + processed_dataset_info['n_num'] + \
           int(processed_dataset_info['is_regression'] and not model_params['is_y_cond'])
    model_params['d_in'] = d_in
    print(f'log: n_cat_emb: {processed_dataset_info["n_cat_emb"]} d_in: {model_params["d_in"]}')
    
    print(model_params)
    model = MLPDiffusion(**model_params)
    model.to(device)

    X_train, y_train = data['X_train'], data['y_train']
    training_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    diffusion = GaussianDiffusion(
        embedding_sizes=processed_dataset_info['embedding_sizes'],
        num_features=model_params['d_in'],
        denoise_fn=model,
        loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        beta_scheduler=scheduler,
        device=device
    )
    diffusion.to(device)
    diffusion.train()

    trainer = Trainer(
        diffusion,
        dataloader,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        device=device
    )
    trainer.run_loop(entity_encoder)

    trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))