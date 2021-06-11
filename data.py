import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
import wandb


def clean_round(array: np.array) -> np.array:
    rem = 0
    res = np.zeros_like(array, dtype=int)
    for i, item in enumerate(array):
        res[i] = round(item+rem)
        rem = item+rem - res[i]
    assert rem < 0.000001
    return res

def generate_indices(labels, parties = 10, classes_in_use = range(10),
                     avg_samples_per_class = 20, concentration = 'iid',
                     balance = True, data_overlap = False):
    if concentration == 'iid':
        concentration = 1e10

    if parties < len(classes_in_use):
        balance = False

    for _ in range(10000):  # dont block infinite when balancing
        dists = np.random.dirichlet(np.repeat(concentration, len(classes_in_use)), parties)
        # rounding would imbalance total samples per party
        num_samples = np.array([clean_round(d) for d in \
                                dists * len(classes_in_use) * avg_samples_per_class])
        class_sum = num_samples.sum(axis=0)
        if not balance \
           or (class_sum.min() > avg_samples_per_class/2 and
               class_sum.max() - class_sum.min() < avg_samples_per_class*parties/2):
            break

    samples_cumsum = np.cumsum(num_samples, axis=0)
    idxs = [np.array([],dtype=int)] * parties
    for n, cls in enumerate(classes_in_use):
        idx = np.nonzero(labels == cls)[0]
        idx = np.random.choice(idx, class_sum[n], replace = data_overlap)
        idx = np.split(idx, samples_cumsum[:-1,n])
        for i in range(parties):
            idxs[i] = np.r_[idxs[i], idx[i]]

    return idxs, num_samples, dists



def generate_dist_indices(targets, classes_in_use = range(10), dists = None):
    if dists is None:
        dists = [[1/len(classes_in_use)] * classes_in_use]
    else:
        assert(np.allclose(np.sum(dists, axis=1), 1))

    counts = np.bincount(targets)
    counts = counts.take(classes_in_use)
    num_samples = [clean_round(d) for d in dists * counts]

    idxs = []
    for num in num_samples:
        combined_idx = np.array([], dtype = np.int16)
        for cls, n in zip(classes_in_use, num):
            all_idx = (targets == cls).nonzero()[0]
            combined_idx = np.r_[combined_idx, all_idx[:n]]
        idxs.append(combined_idx)
    return idxs


def generate_total_indices(targets, classes_in_use = range(10)):
    combined_idx = np.array([], dtype = np.int16)
    for cls in classes_in_use:
        idx = np.where(targets == cls)[0]
        combined_idx = np.r_[combined_idx, idx]
    return combined_idx


def get_idx_artifact_name(cfg):
    return f"p{cfg['parties']}_s{cfg['samples_per_class']}_c{cfg['concentration']}_C{'-'.join(map(str,cfg['subclasses']))}"

def save_idx_to_artifact(cfg, idxs, num_samples, test_idxs):
    idx_artifact_name = get_idx_artifact_name(cfg)
    idx_artifact = wandb.Artifact(idx_artifact_name, type='private_indices',
                                    metadata={
                                        'parties': cfg['parties'],
                                        'samples_per_class': cfg['samples_per_class'],
                                        'subclasses': cfg['subclasses'],
                                        'concentration': cfg['concentration'],
                                        'distributions': num_samples,
                                        'class_total': num_samples.sum(axis=0)})
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
        idxs = np.load(idx_file)
        test_idx_file = idx_artifact.get_path('test_idxs.npy').download()
        test_idxs = np.load(test_idx_file)
        print(f'Private Idx: Use "{idx_artifact_name}" artifact with saved private indices')
        
    except (wandb.CommError, Exception):
        print(f'Private Idx: Create "{idx_artifact_name}" artifact with new random private indices')

        idxs, num_samples, dists = generate_indices(
            targets, cfg['parties'], cfg['subclasses'],
            cfg['samples_per_class'], cfg['concentration'])
        test_idxs = generate_dist_indices(test_targets, cfg['subclasses'], dists)
        idx_artifact = save_idx_to_artifact(cfg, idxs, num_samples, test_idxs)
    try:
        idx_artifact.wait()  # throws execption in offline mode
    except Exception as e:
        pass
    dists = idx_artifact.metadata['distributions']
    total = idx_artifact.metadata['class_total']
    print("party distributions:\n", dists)
    print("class total:\n", total)
    return idxs, test_idxs


def build_private_dls(private_train_data, private_test_data,
                      private_idxs, private_test_idxs,
                      num_public_classes, subclasses,
                      batch_size):
    def collate_fn(batch):
        batch = [ (i[0], num_public_classes + subclasses.index(i[1])) for i in batch]
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


    combined_dl = DataLoader(Subset(private_train_data,
                                    np.concatenate(private_idxs)),
                             batch_size=batch_size,
                             collate_fn=collate_fn, shuffle=True)

    private_sub_test_idx = generate_total_indices(np.array(private_test_data.targets),
                                                  subclasses)
    combined_test_dl = DataLoader(Subset(private_test_data, private_sub_test_idx),
                                  batch_size=batch_size,
                                  collate_fn=collate_fn)

    return private_dls, private_test_dls, combined_dl, combined_test_dl
