from catboost import CatBoostClassifier, CatBoostRegressor
import pickle
import json
import pandas as pd
from sklearn.metrics import classification_report, r2_score
import numpy as np
import os
from sklearn.utils import shuffle
import zero
from pathlib import Path
from pprint import pprint
from lib import calculate_metrics, MetricsReport, TaskType, PredictionType
from lib import load_json, dump_json, tensor2ndarray
from scripts.eval_discriminator import train_discriminator
from scripts.eval_stats import cal_avg_jsd, cal_avg_wd, cal_correlation_diff


def get_catboost_config(ds_name):
    if os.path.exists(f'../tuned_models/catboost/{ds_name}_cv.json'):
        path = f'../tuned_models/catboost/{ds_name}_cv.json'
    else:
        path = f'tuned_models/catboost/{ds_name}_cv.json'
    C = load_json(path)
    return C


def get_npydataset(x_path, y_path, x_num_path=None):
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


def split_dataset(n_cat, dataset):
    X_cat = tensor2ndarray(dataset[:, :n_cat])
    y_train = tensor2ndarray(dataset[:, n_cat])
    X_num = tensor2ndarray(dataset[:, n_cat + 1:])
    X_train = np.hstack([X_cat, X_num])
    return X_train, y_train


def train_catboost(
        raw_data,
        dataset_info,
        ds_name,
        parent_dir,
        preprocessors,
        eval_type='synthetic',
        synthetic_data=None,
        seed=2022,
        params=None,
        get_discriminator_measure=True,
        get_statistical_similarity=True
):
    assert eval_type == 'synthetic' or eval_type == 'real'

    if params is None:
        catboost_config = get_catboost_config(ds_name)
    else:
        catboost_config = params

    if 'cat_features' not in catboost_config:
        catboost_config['cat_features'] = list(range(dataset_info['n_cat']))

    X_test, y_test = tensor2ndarray(raw_data['X_test']), tensor2ndarray(raw_data['y_test'])
    n_columns = X_test.shape[1]

    if eval_type == 'synthetic':
        assert synthetic_data is not None
        if dataset_info['including_y']:
            X_train, y_train = split_dataset(dataset_info['n_cat'], synthetic_data)
        else:
            X_train, y_train = synthetic_data
            X_train, y_train = tensor2ndarray(X_train), tensor2ndarray(y_train)

        X_train = pd.DataFrame(X_train, columns=[i for i in range(n_columns)])

    else:  # eval_type == 'real'
        if dataset_info['including_y']:
            X_train, _ = split_dataset(dataset_info['n_cat'], raw_data['X_train'])
            y_train = tensor2ndarray(raw_data['y_train'])
        else:
            X_train, y_train = tensor2ndarray(raw_data['X_train']), tensor2ndarray(raw_data['y_train'])
        if dataset_info['is_regression'] and not dataset_info['is_y_cond']:
            n_columns += 1
        X_train = pd.DataFrame(X_train, columns=[i for i in range(n_columns)])
        normalizer = preprocessors["normalizer"]
        continuous_cols = [i for i in range(n_columns) if i not in catboost_config['cat_features']]
        X_cat = X_train[catboost_config['cat_features']]
        X_cont = X_train[continuous_cols]
        if dataset_info['is_regression'] and not dataset_info['is_y_cond']:
            X_cont_ = normalizer.inverse_transform(X_cont)
            X_cont = X_cont_[:, :-1]
            n_columns -= 1
        else:
            X_cont = normalizer.inverse_transform(X_cont)

        X_train = pd.DataFrame(np.hstack([X_cat, X_cont]), columns=[i for i in range(n_columns)])

    if dataset_info['including_y']:
        X_val, _ = split_dataset(dataset_info['n_cat'], raw_data['X_val'])
        y_val = tensor2ndarray(raw_data['y_val'])
    else:
        X_val, y_val = tensor2ndarray(raw_data['X_val']), tensor2ndarray(raw_data['y_val'])

    if dataset_info['is_regression'] and not dataset_info['is_y_cond']:
        n_columns += 1
    X_val = pd.DataFrame(X_val, columns=[i for i in range(n_columns)])
    normalizer = preprocessors["normalizer"]
    continuous_cols = [i for i in range(n_columns) if i not in catboost_config['cat_features']]
    X_cat = X_val[catboost_config['cat_features']].values
    X_cont = X_val[continuous_cols].values
    if dataset_info['is_regression'] and not dataset_info['is_y_cond']:
        X_cont_ = normalizer.inverse_transform(X_cont)
        X_cont = X_cont_[:, :-1]
        n_columns -= 1
    else:
        X_cont = normalizer.inverse_transform(X_cont)

    X_val = pd.DataFrame(np.hstack([X_cat, X_cont]), columns=[i for i in range(n_columns)])

    # use the encoder in the preprocessing
    X_test = pd.DataFrame(X_test, columns=[i for i in range(n_columns)])
    X_test_cat = X_test[catboost_config["cat_features"]].values
    cat_encoder = preprocessors['cat_encoder']
    n_col = X_test_cat.shape[1]
    # add category appears in test set
    add_category_func = np.frompyfunc(lambda s, idx: -1 if s not in cat_encoder.categories_[idx] else s, nin=2, nout=1)
    for i in range(n_col):
        X_test_cat[:, i] = add_category_func(X_test_cat[:, i], i)
        if -1 in X_test_cat[:, i]:
            cat_encoder.categories_[i] = np.append(cat_encoder.categories_[i], -1)

    X_test[catboost_config["cat_features"]] = cat_encoder.transform(X_test_cat)

    for col in catboost_config["cat_features"]:
        X_test[col] = X_test[col].astype('str')
        X_val[col] = X_val[col].astype('str')
        X_train[col] = X_train[col].astype('str')

    print(f"{eval_type} train set X and y: ")
    print(X_train.shape, X_train.head(5))
    print(y_train.shape, y_train[:5])
    print("val set X and y: ")
    print(X_val.head(5))
    print(y_val[:5])
    print("test set X and y: ")
    print(X_test.head(5))
    print(y_test[:5])

    if dataset_info['is_regression']:
        model = CatBoostRegressor(
            **catboost_config,
            eval_metric='RMSE',
            random_seed=seed
        )
        predict = model.predict
    else:
        n_classes = dataset_info['n_classes']
        assert n_classes >= 2
        if n_classes > 2:
            is_multiclass = True
        else:
            is_multiclass = False
        model = CatBoostClassifier(
            loss_function="MultiClass" if is_multiclass else "Logloss",
            **catboost_config,
            eval_metric='TotalF1',
            random_seed=seed,
            class_names=[str(i) for i in range(n_classes)] if is_multiclass else ["0", "1"]
        )
        predict = (
            model.predict_proba
            if is_multiclass
            else lambda x: model.predict_proba(x)[:, 1]
        )

    print('log: Catboost training starts.')
    if dataset_info['is_regression']:
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=100
        )
    else:
        model.fit(
            X_train, y_train.astype(np.int8),
            eval_set=(X_val, y_val.astype(np.int8)),
            verbose=100
        )

    predictions_test = predict(X_test)
    print('log: test predictions: ', predictions_test.shape, predictions_test[:10])

    predictions_val = predict(X_val)
    print('log: val predictions: ', predictions_val.shape, predictions_val[:10])

    predictions_train = predict(X_train)
    print('log: train predictions: ', predictions_train.shape, predictions_train[:10])

    if dataset_info['is_regression']:
        task_type = TaskType('regression')
        prediction_type = None
    elif dataset_info['n_classes'] <= 2:
        task_type = TaskType('binclass')
        # prediction_type = PredictionType('logits')
        prediction_type = PredictionType('probs')
    else:
        task_type = TaskType('multiclass')
        prediction_type = PredictionType('probs')

    print(task_type)

    report = {}
    report['dataset'] = ds_name
    report['ml efficiency'] = {'test': calculate_metrics(y_true=y_test,
                                                         y_pred=predictions_test,
                                                         task_type=task_type,
                                                         prediction_type=prediction_type,
                                                         y_info={'std': 1.0} if dataset_info['is_regression'] else {}),

                               'val': calculate_metrics(y_true=y_val,
                                                        y_pred=predictions_val,
                                                        task_type=task_type,
                                                        prediction_type=prediction_type,
                                                        y_info={'std': 1.0} if dataset_info['is_regression'] else {}),

                               'train': calculate_metrics(y_true=y_train,
                                                          y_pred=predictions_train,
                                                          task_type=task_type,
                                                          prediction_type=prediction_type,
                                                          y_info={'std': 1.0} if dataset_info['is_regression'] else {})}

    print(report)

    if task_type == TaskType.REGRESSION:
        score_key = 'rmse'
        score_sign = -1
    else:
        score_key = 'accuracy'
        score_sign = 1
    for part_metrics in report['ml efficiency'].values():
        part_metrics['score'] = score_sign * part_metrics[score_key]

    metrics_report = MetricsReport(report['ml efficiency'], task_type)
    report['ml efficiency'] = metrics_report.print_metrics()

    X_synth, y_synth = X_train.values, y_train
    if eval_type == 'synthetic' and get_discriminator_measure:
        if dataset_info['including_y']:
            X_train, _ = split_dataset(dataset_info['n_cat'], raw_data['X_train'])
        else:
            X_train = tensor2ndarray(raw_data['X_train'])
        if dataset_info['is_regression'] and not dataset_info['is_y_cond']:
            n_columns += 1
        X_train = pd.DataFrame(X_train, columns=[i for i in range(n_columns)])
        normalizer = preprocessors["normalizer"]
        continuous_cols = [i for i in range(n_columns) if i not in catboost_config['cat_features']]
        X_cat = X_train[catboost_config['cat_features']]
        X_cont = X_train[continuous_cols]
        if dataset_info['is_regression'] and not dataset_info['is_y_cond']:
            X_cont_ = normalizer.inverse_transform(X_cont)
            X_cont = X_cont_[:, :-1]
            n_columns -= 1
        else:
            X_cont = normalizer.inverse_transform(X_cont)

        X_train = np.hstack([X_cat, X_cont])
        y_train = tensor2ndarray(raw_data['y_train'])
        y_test = tensor2ndarray(raw_data['y_test'])

        disc_report = train_discriminator(X_train, y_train, X_test.values, y_test, X_synth, y_synth, seed=seed)
        report['discriminator measure'] = disc_report
        print('discriminator results: ')
        print(disc_report)

    if eval_type == 'synthetic' and get_statistical_similarity:
        stats_report = {}

        nomalizer = preprocessors['normalizer']
        if dataset_info['is_regression'] and not dataset_info['is_y_cond']:
            X_train = np.hstack([X_train, y_train.reshape(-1, 1)])
            X_synth = np.hstack([X_synth, y_synth.reshape(-1, 1)])
            X_train[:, dataset_info['n_cat']:] = nomalizer.transform(X_train[:, dataset_info['n_cat']:])
            X_synth[:, dataset_info['n_cat']:] = nomalizer.transform(X_synth[:, dataset_info['n_cat']:])
            X_train = X_train[:, :-1]
            X_synth = X_synth[:, :-1]
        else:
            X_train[:, dataset_info['n_cat']:] = nomalizer.transform(X_train[:, dataset_info['n_cat']:])
            X_synth[:, dataset_info['n_cat']:] = nomalizer.transform(X_synth[:, dataset_info['n_cat']:])

        X_cat = X_train[:, :dataset_info['n_cat']]
        X_cat_synth = X_synth[:, :dataset_info['n_cat']]
        X_num = X_train[:, dataset_info['n_cat']:]
        X_num_synth = X_synth[:, dataset_info['n_cat']:]
        stats_report['avg_jsd'] = cal_avg_jsd(X_cat, X_cat_synth)
        stats_report['avg_wd'] = cal_avg_wd(X_num, X_num_synth)
        X_train = pd.DataFrame(X_train, columns=[i for i in range(n_columns)])
        X_synth = pd.DataFrame(X_synth, columns=[i for i in range(n_columns)])
        stats_report['correlation_diff'] = cal_correlation_diff(X_train, X_synth, catboost_config["cat_features"])
        report['statistical similarity'] = stats_report
        print('statistical similarity: ')
        print(stats_report)

    if parent_dir is not None:
        try:
            dump_json(report, os.path.join(parent_dir, "results_catboost.json"))
        except Exception:
            print(Exception)

    return report
