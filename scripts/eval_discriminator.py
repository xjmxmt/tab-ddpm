import numpy as np
import os
import pickle
import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from lib import tensor2ndarray


def get_dataset(x_path, y_path, x_num_path=None):
    y_train = np.load(y_path, allow_pickle=True)
    if x_num_path is not None:
        x_cat_path = x_path
        X_train_cat = np.load(x_cat_path, allow_pickle=True)
        X_train_num = np.load(x_num_path, allow_pickle=True)
        X_train = np.hstack([X_train_cat, X_train_num])
        return X_train_cat, X_train_num, X_train, y_train
    else:
        X_train = np.load(x_path, allow_pickle=True)
    return X_train, y_train


def cal_metric(metric, y_true, y_pred):
    if metric == 'acc':
        metric = accuracy_score(y_true, y_pred)
    elif metric == 'macro_f1':
        metric = f1_score(y_true, y_pred, average='macro')
    elif metric == 'micro_f1':
        metric = f1_score(y_true, y_pred, average='micro')
    return metric


def train_discriminator(X_train, y_train, X_test, y_test, X_synth, y_synth, seed=42):

    # prepare train dataset
    X_train = np.hstack([X_train, y_train.reshape(-1, 1)])
    print('log: raw train set: ', X_train.shape)

    # # prepare synthetic train dataset
    # assert os.path.exists(os.path.join(parent_dir, 'X_train.npy')) and os.path.exists(os.path.join(parent_dir, 'y_train.npy'))
    # x_path = os.path.join(parent_dir, 'X_train.npy')
    # y_path = os.path.join(parent_dir, 'y_train.npy')
    # X_synth, y_synth = get_dataset(x_path, y_path)

    X_train_synth, X_test_synth, y_train_synth, y_test_synth = train_test_split(X_synth, y_synth, test_size=0.2, random_state=seed, shuffle=True)
    X_train_synth = np.hstack([X_train_synth, y_train_synth.reshape(-1, 1)])
    X_test_synth = np.hstack([X_test_synth, y_test_synth.reshape(-1, 1)])
    print('log: synthetic train set: ', X_train_synth.shape)

    # create discriminator train set
    num_train = min(len(X_train), len(X_train_synth))
    shuffle_idx = np.random.permutation(np.arange(num_train*2))
    X_train_disc = np.vstack([X_train[:num_train, :], X_train_synth[:num_train, :]])[shuffle_idx]
    y_train_disc = np.vstack([np.ones((num_train, 1)), np.zeros((num_train, 1))]).reshape(-1)[shuffle_idx]
    y_train_disc = y_train_disc.astype(int)
    print(f'log: num of train samples: {num_train*2}')

    print(X_train_disc.shape)
    print(X_train_disc[:5])
    print(y_train_disc.shape)
    print(y_train_disc[:5])

    # prepare train dataset
    X_test = np.hstack([X_test, y_test.reshape(-1, 1)])
    print('log: raw test set: ', X_test.shape)
    print('log: synthetic test set: ', X_test_synth.shape)

    # prepare synthetic test dataset
    num_test = min(len(X_test), len(X_test_synth))
    shuffle_idx = np.random.permutation(np.arange(num_test*2))
    X_test_disc = np.vstack([X_test[:num_test, :], X_test_synth[:num_test, :]])[shuffle_idx]
    y_test_disc = np.vstack([np.ones((num_test, 1)), np.zeros((num_test, 1))]).reshape(-1)[shuffle_idx]
    y_test_disc = y_test_disc.astype(int)
    print(f'log: num of test samples: {num_test*2}')

    print(X_test_disc.shape)
    print(X_test_disc[:5])
    print(y_test_disc.shape)
    print(y_test_disc[:5])

    # hyperparameter tuning
    model = RandomForestClassifier(random_state=seed, criterion='gini', max_features='auto')
    grid_param = {
        "n_estimators": [3, 5, 10, 50, 100],
        "max_depth": range(2, 5, 1),
        "min_samples_leaf": range(1, 5, 1),
        "min_samples_split": range(2, 5, 1),
    }

    grid_search = GridSearchCV(estimator=model, param_grid=grid_param, cv=2, n_jobs=-1, verbose=-1)
    grid_search.fit(X_train_disc, y_train_disc)
    print('best params: ', grid_search.best_params_)
    print('best training score: ', grid_search.best_score_)
    clf = grid_search.best_estimator_

    # clf = LogisticRegression(random_state=seed)
    # clf.fit(X_train_disc, y_train_disc)

    clf.fit(X_train_disc, y_train_disc)
    y_pred_train = clf.predict(X_train_disc)
    y_pred = clf.predict(X_test_disc)

    report = {
        'train': {'acc': cal_metric('acc', y_train_disc, y_pred_train),
                  'macro f1': cal_metric('macro_f1', y_train_disc, y_pred_train),
                  'micro f1': cal_metric('micro_f1', y_train_disc, y_pred_train)},

        'test': {'acc': cal_metric('acc', y_test_disc, y_pred),
                 'macro f1': cal_metric('macro_f1', y_test_disc, y_pred),
                 'micro f1': cal_metric('micro_f1', y_test_disc, y_pred)}
    }

    return report
