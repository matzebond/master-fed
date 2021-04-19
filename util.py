import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt


def example(model, data, classes, inst):
    model.eval()
    x, y = data[inst]
    with torch.no_grad():
        pred = model(x.to(device).unsqueeze_(0))
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

        plt.imshow(transforms.ToPILImage()(x))
        plt.show()
        
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def example_batch(model, batch, classes):
    model.eval()
    x, y = batch
    with torch.no_grad():
        logits = model(x.to(device))
        pred = logits.argmax(-1)
        correct = (pred == y.to(device)).sum()
        
        imshow(torchvision.utils.make_grid(x))

        print(f'Predicted: \n{", ".join(["%5s" % classes[i] for i in pred])}')
        print(f'Actual: \n{", ".join(["%5s" % classes[i] for i in y])}')
        print(f'Acc: {correct/len(y)}' )
        #plt.imshow(transforms.ToPILImage()(x))
        #plt.show()



def show_img(data, classes, idx):
    x, y = data[idx]
    print(f'Class: "{classes[y]}"')
    plt.imshow(transforms.ToPILImage()(x))
    plt.show()
    
def show_tensor(tensor, label, classes):
    print(f'Class: "{classes[label]}"')
    plt.imshow(transforms.ToPILImage()(tensor))
    plt.show()

def show_batch(batch, classes, idx):
    x, y = batch
    print(f'Class: "{classes[y[idx]]}"')
    plt.imshow(transforms.ToPILImage()(x[idx]))
    plt.show()



def dataloader_details(dataloader, class_names=[]):
    print(len(dataloader))
    print(len(dataloader.dataset))

    pass
    
def dataset_details(dataset, class_names):
    labels = {}
    for data,label in dataset:
        name = class_names[label]
        if name in labels:
            labels[name] += 1
        else:
            labels[name] = 1
    return labels

def dataset_split(dataset):
    split = {}
    for num, (data,label) in enumerate(dataset):
        if label in split:
            split[label].append(data)
        else:
            split[label] = [data]
    return split



def generate_bal_private_data(X, y, N_parties = 10, classes_in_use = range(11), 
                              N_samples_per_class = 20, data_overlap = False):
    """
    Input: 
    -- N_parties : int, number of collaboraters in this activity;
    -- classes_in_use: array or generator, the classes of EMNIST-letters dataset 
    (0 <= y <= 25) to be used as private data; 
    -- N_sample_per_class: int, the number of private data points of each class for each party
    
    return: 
    
    """
    priv_data = [None] * N_parties
    combined_idx = np.array([], dtype = np.int16)
    for cls in classes_in_use:
        idx = np.where(y == cls)[0]
        idx = np.random.choice(idx, N_samples_per_class * N_parties, 
                               replace = data_overlap)
        combined_idx = np.r_[combined_idx, idx]
        for i in range(N_parties):           
            idx_tmp = idx[i * N_samples_per_class : (i + 1)*N_samples_per_class]
            if priv_data[i] is None:
                priv_data[i] = {}
                priv_data[i]["idx"] = idx_tmp
                priv_data[i]["X"] = X[idx_tmp]
                priv_data[i]["y"] = y[idx_tmp]
            else:
                priv_data[i]["idx"] = np.r_[priv_data[i]["idx"], idx_tmp]
                priv_data[i]["X"] = np.vstack([priv_data[i]["X"], X[idx_tmp]])
                priv_data[i]["y"] = np.r_[priv_data[i]["y"], y[idx_tmp]]

    total_priv_data = {}
    total_priv_data["idx"] = combined_idx
    total_priv_data["X"] = X[combined_idx]
    total_priv_data["y"] = y[combined_idx]
    return priv_data, total_priv_data

def generate_partial_data(X, y, classes_in_use = "all", verbose = False):
    if classes_in_use == "all":
        idx = np.ones_like(y, dtype = bool)
    else:
        idx = [y == i for i in classes_in_use]
        idx = np.any(idx, axis = 0)
    X_incomplete, y_incomplete = X[idx], y[idx]
    if verbose == True:
        print("X shape :", X_incomplete.shape)
        print("y shape :", y_incomplete.shape)
    return X_incomplete, y_incomplete



# adapted from generate_bal_private_data to use with torch.dataset.Subset
def generate_partial_indices(y, N_parties = 10, classes_in_use = range(11), 
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

def generate_total_indices(y, classes_in_use = range(11)):
    combined_idx = np.array([], dtype = np.int16)
    for cls in classes_in_use:
        idx = np.where(y == cls)[0]
        combined_idx = np.r_[combined_idx, idx]
    return combined_idx




# save model (after public training)
def save_model_artifact(stage):
    model_artifact = wandb.Artifact(stage, type='model')

    for num, m in enumerate(cfg.model_mapping):
        path = f"models/{stage}_model-{m}.pth"
        torch.save(models[num].state_dict(), path)
        model_artifact.add_file(path)

    main_run.log_artifact(model_artifact)
    print(f'Model: Stale "{stage}" model which is version {model_artifact.version}')
    return model_artifact


def load_model_artifact(stage, version="latest", logging=True):
    model_artifact = main_run.use_artifact(f"{stage}:{version}", type='model')
    model_path = Path(model_artifact.download())
    print(f'Model: Use "{stage}" model which is version {model_artifact.version}')

    # load model from saved state after public training
    for num, m in enumerate(cfg.model_mapping):
        gstep[num] = 0
        model = models[num]
        path = model_path / f"{stage}_model-{m}.pth"
        model.load_state_dict(torch.load(path))

        loss_fn = nn.CrossEntropyLoss()
        print(f"\n-------------------------------\n{stage}: Model-{num} Load\n-------------------------------")
        print(path)
        public_loss, public_acc = test(public_test_dl, model, loss_fn)
        print(f"public Test Error:  Accuracy: {public_acc:>0.4f}%, Avg loss: {public_loss:>8f}")
        private_loss, private_acc = test(private_sub_test_dl, model, loss_fn)
        print(f"private Test Error: Accuracy: {private_acc:>0.4f}%, Avg loss: {private_loss:>8f}")
        
        if logging:
            writer.add_scalar(f"model-{num}/public_loss", public_loss, gstep[num])
            writer.add_scalar(f"model-{num}/public_acc", public_acc, gstep[num])
            writer.add_scalar(f"model-{num}/test_loss", private_loss, gstep[num])
            writer.add_scalar(f"model-{num}/test_acc", private_acc, gstep[num])

            writer.add_scalar(f"model-{num}/coarse_public_loss", public_loss, 0)
            writer.add_scalar(f"model-{num}/coarse_public_acc", public_acc, 0)
            writer.add_scalar(f"model-{num}/coarse_test_loss", private_loss, 0)
            writer.add_scalar(f"model-{num}/coarse_test_acc", private_acc, 0)

            gstep[num] = 1
    return model_artifact
