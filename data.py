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
        p = np.random.dirichlet(np.repeat(concentration, len(classes_in_use)), parties)
        # rounding would imbalance total samples per party
        priv_num = np.array(list(map(clean_round,
                                    p * len(classes_in_use) * avg_samples_per_class)))
        priv_idx = [np.array([],dtype=int)] * parties
        priv_cumsum = np.cumsum(priv_num, axis=0)
        priv_sum = priv_num.sum(axis=0)
        if not balance or (priv_sum.min() > avg_samples_per_class/2 and
                           priv_sum.max() - priv_sum.min() < avg_samples_per_class*parties/2):
            break

    for i, cls in enumerate(classes_in_use):
        idx = np.nonzero(labels == cls)[0]
        idx = np.random.choice(idx, priv_sum[i], replace = data_overlap)
        idx = np.split(idx, priv_cumsum[:-1,i])
        for i in range(parties):
            priv_idx[i] = np.r_[priv_idx[i], idx[i]]

    return priv_idx, priv_num


def generate_total_indices(y, classes_in_use = range(10)):
    combined_idx = np.array([], dtype = np.int16)
    for cls in classes_in_use:
        idx = np.where(y == cls)[0]
        combined_idx = np.r_[combined_idx, idx]
    return combined_idx


def load_idx_from_artifact(targets, parties, subclasses, samples_per_class, concentration):
    idx_artifact_name = f"p{parties}_s{samples_per_class}_c{concentration}_C{'-'.join(map(str,subclasses))}"
    try:
        idx_artifact = wandb.use_artifact(f"{idx_artifact_name}:latest",
                                          type='private_indices')
        # artifact_dir = idx_artifact.download()
        idx_file = idx_artifact.get_path('private_partial_train_idx.npy').download()
        idxs = np.load(idx_file)
        print(f'Private Idx: Use "{idx_artifact_name}" artifact with saved private indices')
        
    except (wandb.CommError, Exception):
        print(f'Private Idx: Create "{idx_artifact_name}" artifact with new random private indices')
        idxs, idxs_num = generate_indices(targets, parties, subclasses, samples_per_class, concentration)
        idx_artifact = wandb.Artifact(idx_artifact_name, type='private_indices',
                                      metadata={'parties': parties,
                                                'samples_per_class': samples_per_class,
                                                'subclasses': subclasses,
                                                'concentration': concentration,
                                                'distributions': idxs_num})
        with idx_artifact.new_file('private_partial_train_idx.npy', 'xb') as f:
            np.save(f, idxs)
        wandb.log_artifact(idx_artifact)
    try:
        idx_artifact.wait()  # throws execption in offline mode
    except Exception as e:
        pass
    print("class distributions:\n", idx_artifact.metadata['distributions'])
    return idxs[:parties]


def build_private_dls(private_train_data, private_test_data,
                      num_public_classes, subclasses,
                      private_idx, batch_size):
    def collate_fn(x):
        x = list(map(lambda i: (i[0],num_public_classes+subclasses.index(i[1])), x))
        return torch.utils.data.dataloader._utils.collate.default_collate(x)

    private_partial_dls = []

    for idx in private_idx:
        dl = DataLoader(Subset(private_train_data, idx),
                        batch_size=batch_size,
                        collate_fn=collate_fn, shuffle=True)
        private_partial_dls.append(dl)


    private_combined_dl = DataLoader(Subset(private_train_data,
                                            np.concatenate(private_idx)),
                                           batch_size=batch_size,
                                           collate_fn=collate_fn, shuffle=True)

    private_sub_test_idx = generate_total_indices(np.array(private_test_data.targets),
                                                  subclasses)
    private_test_dl = DataLoader(Subset(private_test_data, private_sub_test_idx),
                                 batch_size=batch_size,
                                 collate_fn=collate_fn)

    return private_partial_dls, private_combined_dl, private_test_dl
