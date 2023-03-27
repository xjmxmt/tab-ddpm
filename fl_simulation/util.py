import numpy as np
from lib.util import tensor2ndarray


def dirichlet_split_noniid(train_labels, n_clients, alpha=1.0, seed=42):
    np.random.seed(seed)
    n_classes = train_labels.max()+1

    train_labels = tensor2ndarray(train_labels)
    train_labels = train_labels.astype(np.long).reshape(-1, )

    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]

    flag = False
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    client_idcs = [[] for _ in range(n_clients)]

    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]
            if len(idcs) == 0:
                flag = True

    while flag:
        flag = False
        label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
        client_idcs = [[] for _ in range(n_clients)]

        for c, fracs in zip(class_idcs, label_distribution):
            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                client_idcs[i] += [idcs]
                if len(idcs) == 0:
                    flag = True

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs


def get_data_splits(dataset_length, num_clients, type='uneven'):
    data_split_dict = {}
    if type == 'average':
        # average split
        split = dataset_length // num_clients
        start = 0
        for i in range(num_clients-1):
            data_split_dict[str(i)] = (start, start + split)
            start += split
        data_split_dict[str(num_clients-1)] = (start, dataset_length)
    elif type == 'uneven':
        splits = np.random.randint(dataset_length // num_clients // 4, dataset_length // num_clients * 4, num_clients)

        while sum(splits) > dataset_length:
            splits = np.random.randint(dataset_length // num_clients // 4, dataset_length // num_clients * 4, num_clients)
        if sum(splits) < dataset_length:
            left = dataset_length - sum(splits)
            splits += left // num_clients

        start = 0
        for i in range(num_clients-1):
            data_split_dict[str(i)] = (start, start + splits[i])
            start += splits[i]
        data_split_dict[str(num_clients - 1)] = (start, dataset_length)
    return data_split_dict


if __name__ == '__main__':
    train_labels = np.array([0, 0, 0, 1, 1, 1, 1])
    print(dirichlet_split_noniid(train_labels, 3))
