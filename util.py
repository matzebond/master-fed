import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda, Compose
from tensorboard_reducer import load_tb_events, write_tb_events, reduce_events
import matplotlib.pyplot as plt
import wandb
from collections import defaultdict
from pathlib import Path
import os
import shutil
from typing import List


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
    split = defaultdict(list)
    for num, (data,label) in enumerate(dataset):
        split[label].append(data)
    return split



def save_models_to_artifact(cfg, workers, info, stage):
    model_artifact = wandb.Artifact(stage, type='model')

    for rank, worker in enumerate(workers):
        model_artifact.add_file(f"{worker.cfg['path']}/{stage}.pth",
                                f"r{rank}-m{cfg['model_mapping'][rank]}-{stage}.pth")
        model_artifact.add_file(f"{worker.cfg['path']}/{stage}_optim.pth",
                                f"r{rank}-m{cfg['model_mapping'][rank]}-{stage}_optim.pth")

    wandb.log_artifact(model_artifact)
    try:
        print(f'Model: Save "{stage}" models as version {model_artifact.version}')
    except Exception as e:
        print(f'Model: Save "{stage}" models in offline mode')
    return model_artifact

def load_models_from_artifact(cfg, workers, stage, version="latest"):
    model_artifact = wandb.use_artifact(f"{stage}:{version}", type='model')
    artifact_path = Path(model_artifact.download())
    print(f'Model: Use "{stage}" model from version {model_artifact.version}')

    for rank, worker in enumerate(workers):
        p = Path.cwd() / artifact_path / f"r{rank}-m{cfg['model_mapping'][rank]}-{stage}.pth"
        (worker.cfg['path'] / f"{stage}.pth").symlink_to(p)

        p = Path.cwd() / artifact_path / f"r{rank}-m{cfg['model_mapping'][rank]}-{stage}_optim.pth"
        (worker.cfg['path'] / f"{stage}_optim.pth").symlink_to(p)

    # wandb.run.summary["init_public/acc"] = np.average(acc)
    # wandb.run.summary["init_public/loss"] = np.average(loss)
    return model_artifact



def merge_tb_files(input_dirs: List[Path]):
    comp_files = []
    for d in input_dirs:
        event_files = list(d.glob("events.out.tfevents*"))
        comp_file = d / "events.out.tfevents.complete"
        with open(comp_file, "xb") as compfile:
            for f in event_files:
                with open(f, "rb") as partfile:
                    shutil.copyfileobj(partfile, compfile)
                # os.remove(f)
        comp_files.append(str(comp_file))

def reduce_tb_events(indirs_glob, outdir,
                     overwrite = False,
                     reduce_ops = ["mean", "min", "max"]):
    events_dict = load_tb_events(indirs_glob)
    reduced_events = reduce_events(events_dict, reduce_ops)
    write_tb_events(reduced_events, outdir, overwrite)
