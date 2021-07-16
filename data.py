from itertools import cycle
import numpy as np
import torch
import torch.utils.data.dataloader
import random
from torch.utils.data import DataLoader, Subset, TensorDataset
import wandb
import util

# deprecated
def clean_round(array: np.ndarray) -> np.ndarray:
    rem = 0
    res = np.zeros_like(array, dtype=int)
    for i, item in enumerate(array):
        res[i] = round(item+rem)
        rem = item+rem - res[i]
    # assert rem < 0.000001
    return res

def partition_data(labels, parties = 10, classes_in_use = range(10),
                   normalize = "class", samples = None, concentration = 'iid',
                   data_overlap = False, balance = True):
    if concentration == 'iid':
        concentration = 1e10
    elif concentration == 'single_class':
        concentration = 1e-3

    if normalize == 'moon':
        return partition_moon(labels, parties, classes_in_use, samples, concentration)

    if parties < len(classes_in_use):
        print('Too few parties. No balancing of the indices.')
        balance = False

    dists = np.array([])
    for _ in range(10000):  # dont block infinite when balancing
        if concentration > 0.005:
            dists = np.random.dirichlet(np.repeat(concentration, len(classes_in_use)),
                                        parties)
        else: # below 0.005 numpys dirichlet gives NaN rows
            dists = np.zeros((parties, len(classes_in_use)))
            for p, c in zip(range(parties),
                            cycle(np.random.permutation(classes_in_use))):
                dists[p][classes_in_use.index(c)] = 1

        class_sum = dists.sum(axis=0)
        if not balance or class_sum.max() - class_sum.min() < class_sum.mean():
            break
    else:
        print("Could not balance the distribution.")

    cumsums = np.cumsum(dists, axis=0)
    idxs = [np.array([],dtype=int)] * parties
    counts = np.zeros((parties, len(classes_in_use)),dtype=int)

    for n, cls in enumerate(classes_in_use):
        idx = np.nonzero(labels == cls)[0]
        amount = samples or len(idx)
        if normalize == 'class':
            cumsums[:,n] /= cumsums[-1,n]
            dists[1:,n] = np.diff(cumsums[:,n])
            splits = (cumsums[:,n] * amount).round().astype(int)
            idx = np.split(idx, splits)[:-1]
        elif normalize == 'party':
            splits = (cumsums[:,n] * amount).round().astype(int)
            idx = np.random.choice(idx, splits[-1], replace=data_overlap)
            idx = np.split(idx, splits)[:-1]
        for i in range(parties):
            idxs[i] = np.r_[idxs[i], idx[i]]
        counts[:,n] = [ len(i) for i in idx ]

    return idxs, counts, dists


def partition_moon(labels, parties = 10, classes_in_use = range(10),
                   samples = None, concentration = 0.5):
    min_size = 0
    min_require_size = 1
    N = labels.shape[0]
    idx_arr = []
    idx_batch = [[] for _ in range(parties)]

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(parties)]
        for k in classes_in_use:
            idx_k = np.where(labels == k)[0]
            if samples:
                idx_k = idx_k[:samples]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(concentration, parties))
            proportions = np.array([p * (len(idx_j) < N / parties) for p, idx_j \
                                    in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx \
                            in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            print(min_size)
            # if K == 2 and n_parties <= 10:
            #     if np.min(proportions) < 200:
            #         min_size = 0
            #         break


    for j in range(parties):
        np.random.shuffle(idx_batch[j])
        idx_arr.append(np.array(idx_batch[j]))

    print([x.shape for x in idx_arr])

    all_classes = np.unique(labels).shape[0]

    counts = np.zeros((parties, len(classes_in_use)), dtype=int)
    for i,idx in enumerate(idx_arr):
        tmp = np.bincount(labels[idx], minlength=all_classes)
        for j, c in enumerate(classes_in_use):
            counts[i][j] = tmp[c]

    dists = (np.array([(counts[i] / s) for i,s in enumerate(counts.sum(axis=1))]))
    return idx_arr, counts, dists


def partition_by_dist(targets, classes_in_use = range(10), dists = None):
    if dists is None:
        dists = [[1/len(classes_in_use)] * classes_in_use]

    counts = np.bincount(targets)
    counts = counts.take(classes_in_use)
    num_samples = [d.round().astype(int) for d in dists * counts]

    idxs = []
    for num in num_samples:
        combined_idx = np.array([], dtype = np.int16)
        for cls, n in zip(classes_in_use, num):
            all_idx = (targets == cls).nonzero()[0]
            combined_idx = np.r_[combined_idx, all_idx[:n]]
        idxs.append(combined_idx)
    return idxs


def generate_total_indices(targets, classes_in_use = range(10)) -> np.ndarray:
    if classes_in_use is None:
        return targets

    combined_idx = np.array([], dtype = np.int16)
    for cls in classes_in_use:
        idx = np.where(targets == cls)[0]
        combined_idx = np.r_[combined_idx, idx]
    return combined_idx


def get_idx_artifact_name(cfg):
    if cfg['partition_normalize'] == 'party':
        name = f"p{cfg['parties']}_s{cfg['samples']}_c{cfg['concentration']}_C{'-'.join(map(str,cfg['classes']))}"
    else:
        name = f"p{cfg['parties']}_n{cfg['partition_normalize']}_s{cfg['samples']}_c{cfg['concentration']}_C{'-'.join(map(str,cfg['classes']))}"

    if cfg['seed'] != 0:
        name += '_' + str(cfg['seed'])

    return name


def save_idx_to_artifact(cfg, idxs, counts, test_idxs):
    idx_artifact_name = get_idx_artifact_name(cfg)
    idx_artifact = wandb.Artifact(idx_artifact_name, type='private_indices',
                                    metadata={
                                        'parties': cfg['parties'],
                                        'normalize': cfg['partition_normalize'],
                                        'samples': cfg['samples'],
                                        'classes': cfg['classes'],
                                        'concentration': cfg['concentration'],
                                        'distributions': counts,
                                        'class_total': counts.sum(axis=0),
                                        'party_total': counts.sum(axis=1)})
    with idx_artifact.new_file('idxs.npy', 'xb') as f:
        np.save(f, idxs)
    with idx_artifact.new_file('test_idxs.npy', 'xb') as f:
        np.save(f, test_idxs)
    wandb.log_artifact(idx_artifact)
    return idx_artifact


def load_idx_from_artifact(cfg, targets, test_targets):
    idx_artifact_name = get_idx_artifact_name(cfg)
    try:
        idx_artifact = wandb.use_artifact(f"{idx_artifact_name}:latest",
                                          type='private_indices')
        # artifact_dir = idx_artifact.download()
        idx_file = idx_artifact.get_path('idxs.npy').download()
        idxs = np.load(idx_file, allow_pickle=True)
        test_idx_file = idx_artifact.get_path('test_idxs.npy').download()
        test_idxs = np.load(test_idx_file, allow_pickle=True)
        print(f'Private Idx: Use "{idx_artifact_name}" artifact with saved private indices')
        
    except (wandb.CommError, AttributeError) as e:
        print(e)
        print(f'Private Idx: Create "{idx_artifact_name}" artifact with new random private indices')

        idxs, counts, dists = partition_data(
            targets, cfg['parties'], cfg['classes'],
            cfg['partition_normalize'], cfg['samples'],
            cfg['concentration'], cfg['partition_overlap'])


        test_idxs = partition_by_dist(test_targets, cfg['classes'], dists)
        idx_artifact = save_idx_to_artifact(cfg, idxs, counts, test_idxs)
    except Exception as e:
        raise e

    try:
        idx_artifact.wait()  # throws execption in offline mode
    except Exception as e:
        pass

    try:
        dists = idx_artifact.metadata['distributions']
        print("party distributions:\n", dists)
        total_party = idx_artifact.metadata['party_total']
        print("party total:\n", total_party)
        total_class = idx_artifact.metadata['class_total']
        print("class total:\n", total_class)
    except Exception as e:
        pass

    return idxs, test_idxs


def build_private_dls(private_train_data, private_test_data,
                      private_idxs, private_test_idxs,
                      classes, batch_size):
    collate_fn = torch.utils.data.dataloader._utils.collate.default_collate
    if len(classes) != len(private_train_data.classes):
        def collate_fn(batch):
            batch = [ (i[0], classes.index(i[1])) for i in batch ]
            return torch.utils.data.dataloader._utils.collate.default_collate(batch)

    private_dls = []
    private_test_dls = []

    for idx, test_idx in zip(private_idxs, private_test_idxs):
        dl = DataLoader(Subset(private_train_data, idx),
                        batch_size=batch_size,
                        collate_fn=collate_fn, shuffle=True)
        private_dls.append(dl)
        test_dl = DataLoader(Subset(private_test_data, test_idx),
                             batch_size=batch_size,
                             collate_fn=collate_fn)
        private_test_dls.append(test_dl)


    combined_idx = np.concatenate(private_idxs)
    combined_dl = DataLoader(Subset(private_train_data, combined_idx),
                             batch_size=batch_size,
                             collate_fn=collate_fn, shuffle=True)

    private_sub_test_idx = generate_total_indices(np.array(private_test_data.targets),
                                                  classes)
    combined_test_dl = DataLoader(Subset(private_test_data, private_sub_test_idx),
                                  batch_size=batch_size,
                                  collate_fn=collate_fn)

    return private_dls, private_test_dls, combined_dl, combined_test_dl


# deprecated
def get_moon_indices(dataset='CIFAR10', parties = 10, concentration = 0.5, seed=0):
    with util.add_path('../master_moon'):
        import utils as moon_utils

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        random.seed(seed)

        res = moon_utils.partition_data(dataset.lower(), "data", None, "noniid", parties, beta=concentration)

    labels = res[1]
    classes = np.unique(labels)
    idx_map = res[4]

    idxs = [np.array(idx_map[i]) for i in idx_map]

    counts = np.zeros((parties, len(classes)), dtype=int)
    for i,idx in enumerate(idxs):
        counts[i] = np.bincount(labels[idx], minlength=10)

    dists = (np.array([(counts[i] / s) for i,s in enumerate(counts.sum(axis=1))]))
    return idxs, counts, dists
