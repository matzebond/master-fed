import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

# adapted from generate_bal_private_data to use with torch.dataset.Subset
def generate_partial_indices(y, N_parties = 10, classes_in_use = range(10), 
                             N_samples_per_class = 20, data_overlap = False):
    priv_idx = [None] * N_parties
    for cls in classes_in_use:
        idx = np.where(y == cls)[0]
        idx = np.random.choice(idx, N_samples_per_class * N_parties, 
                               replace = data_overlap)
        for i in range(N_parties):           
            idx_tmp = idx[i * N_samples_per_class : (i + 1)*N_samples_per_class]
            if priv_idx[i] is None:
                priv_idx[i] = idx_tmp
            else:
                priv_idx[i] = np.r_[priv_idx[i], idx_tmp]
    return priv_idx


def generate_total_indices(y, classes_in_use = range(10)):
    combined_idx = np.array([], dtype = np.int16)
    for cls in classes_in_use:
        idx = np.where(y == cls)[0]
        combined_idx = np.r_[combined_idx, idx]
    return combined_idx


def load_idx_from_artifact(targets, parties, subclasses, samples_per_class):
    import wandb
    
    idx_artifact_name = f"private_idx_p{parties}_s{samples_per_class}_c{'-'.join(map(str,subclasses))}"

    try:
        idx_artifact = wandb.use_artifact(f"{idx_artifact_name}:latest",
                                          type='private_data')
        # artifact_dir = idx_artifact.download()
        idx_file = idx_artifact.get_path('private_partial_train_idx.npy').download()
        print(idx_file)
        idxs = np.load(idx_file)
        print(f'Private Idx: Use "{idx_artifact_name}" artifact with saved private indices')
        
    except (wandb.CommError, Exception):
        print(f'Private Idx: Create "{idx_artifact_name}" artifact with new random private indices')
        idx_artifact = wandb.Artifact(idx_artifact_name, type='private_data')
    
        idxs = generate_partial_indices(targets, parties, subclasses, samples_per_class)
        np.save('private_partial_train_idx.npy', idxs)
        idx_artifact.add_file('private_partial_train_idx.npy')
        wandb.log_artifact(idx_artifact)
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
