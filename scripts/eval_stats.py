from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import numpy as np
from dython.nominal import associations


def cal_avg_jsd(X_cat: np.ndarray, X_cat_synth: np.ndarray):
    n_cols = X_cat.shape[1]
    scores = []
    for i in range(n_cols):
        col = X_cat[:, i]
        indices, dist = np.unique(col, return_counts=True)
        col_synth = X_cat_synth[:, i]
        indices_synth, dist_synth = np.unique(col_synth, return_counts=True)
        indices = indices.astype(np.float)
        indices_synth = indices_synth.astype(np.float)

        if len(indices_synth) < len(indices):
            tmp = []
            ii = 0
            for v in indices:
                if v not in indices_synth:
                    tmp.append(0)
                else:
                    tmp.append(dist_synth[ii])
                    ii += 1
            dist_synth = np.array(tmp)
        elif len(indices_synth) > len(indices):
            tmp = []
            tmp_synth = []
            indices_set = set.intersection(set(indices), set(indices_synth))
            for v in indices_set:
                ii1 = np.where(indices == v)[0]
                ii2 = np.where(indices_synth == v)[0]
                tmp.append(dist[ii1])
                tmp_synth.append(dist_synth[ii2])
            dist = np.array(tmp)
            dist_synth = np.array(tmp_synth)

        print('indices: ')
        print(len(indices_synth), indices_synth)
        print(len(indices), indices)

        score = jensenshannon(dist, dist_synth) ** 2
        scores.append(score)

    result = np.mean(scores, keepdims=False)
    if isinstance(result, np.ndarray):
        result = result[0]

    return result


def cal_avg_wd(X_num, X_num_synth):
    # # sampling to get the same length
    # n_samples = X_num.shape[0]
    # indices = np.random.choice(X_num_synth.shape[0], n_samples, replace=False)
    # X_num_synth = X_num_synth[indices]

    print(X_num[:10])
    print(X_num_synth[:10])

    n_cols = X_num.shape[1]
    scores = []
    for i in range(n_cols):
        col = X_num[:, i]
        col_synth = X_num_synth[:, i]
        score = wasserstein_distance(col, col_synth)
        print('wd: ', score)
        scores.append(score)
    return np.mean(scores, keepdims=False)


def cal_correlation_diff(X, X_synth, categorical_cols):
    corr_X = associations(X, nominal_columns=categorical_cols, theil_u=True, plot=False)
    corr_X_synth = associations(X_synth, nominal_columns=categorical_cols, theil_u=True, plot=False)
    corr_diff = corr_X['corr'].values - corr_X_synth['corr'].values
    n_cat = len(categorical_cols)
    total_indices = []
    cat_cat = []
    for i in range(n_cat):
        for j in range(i+1, n_cat):
            cat_cat.append(abs(corr_diff[i][j]))
            total_indices.append([i, j])
    cat_cat = np.mean(cat_cat, keepdims=False)
    num_num = []
    for i in range(n_cat, X.shape[1]):
        for j in range(i+1, X.shape[1]):
            num_num.append(abs(corr_diff[i][j]))
            total_indices.append([i, j])
    num_num = np.mean(num_num, keepdims=False)
    num_cat = []
    for i in range(n_cat):
        for j in range(n_cat, X.shape[1]):
            num_cat.append(abs(corr_diff[i][j]))
            total_indices.append([i, j])
    # print(sorted(total_indices, key=lambda i: i[0]))
    num_cat = np.mean(num_cat, keepdims=False)
    return (cat_cat + num_num + num_cat) / 3