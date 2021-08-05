import os
import sys
import shutil
from pathlib import Path
from collections import defaultdict
from typing import (
    List, Callable, Optional, Dict, Sequence, Iterable,
    Union, Tuple, Mapping, Any, Set,
    Generator, TypeVar   
)

import random
import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda, Compose
from tensorboard_reducer import load_tb_events, write_tb_events, reduce_events
from ignite.utils import convert_tensor
import matplotlib.pyplot as plt
import wandb


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

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



def save_models_to_artifact(cfg, workers, stage, metadata,
                            filename=None, type="model"):
    if filename is None: filename = stage
    model_artifact = wandb.Artifact(stage, type=type,
                                    metadata=metadata)

    for worker in workers:
        model_artifact.add_file(
            f"{worker.cfg['tmp']}/{filename}.pth",
            f"{worker.cfg['rank']}-v{worker.cfg['model_variant']}-m{worker.cfg['model_mapping']}-{stage}.pth")
        model_artifact.add_file(
            f"{worker.cfg['tmp']}/{filename}_optim.pth",
            f"{worker.cfg['rank']}-v{worker.cfg['model_variant']}-m{worker.cfg['model_mapping']}-{stage}_optim.pth")

    wandb.log_artifact(model_artifact)
    try:
        model_artifact.wait()  # throws execption in offline mode
        print(f'Model: Save "{stage}" models as version {model_artifact.version}')
    except Exception as e:
        print(f'Model: Save "{stage}" models in offline mode')
    return model_artifact

def load_models_from_artifact(cfg, workers, stage, version="latest",
                              filename=None, type="model"):
    if filename is None: filename = stage
    model_artifact = wandb.use_artifact(f"{stage}:{version}", type=type)
    artifact_path = Path(model_artifact.download())
    print(f'Model: Use "{stage}" model from version {model_artifact.version}')

    for worker in workers:
        p = Path.cwd() / artifact_path / f"{worker.cfg['rank']}-v{worker.cfg['model_variant']}-m{worker.cfg['model_mapping']}-{stage}.pth"
        (worker.cfg['tmp'] / f"{filename}.pth").symlink_to(p)
        worker.model.load_state_dict(torch.load(p))

        p = Path.cwd() / artifact_path / f"{worker.cfg['rank']}-v{worker.cfg['model_variant']}-m{worker.cfg['model_mapping']}-{stage}_optim.pth"
        (worker.cfg['tmp'] / f"{filename}_optim.pth").symlink_to(p)


    wandb.run.summary[f"{stage}/acc"] = model_artifact.metadata['acc']
    wandb.run.summary[f"{stage}/loss"] = model_artifact.metadata['loss']
    return model_artifact, model_artifact.metadata



def concat_tb_files(input_dirs: List[Path]):
    comp_files = []
    for d in input_dirs:
        event_files = sorted(d.glob("events.out.tfevents*"))
        comp_file = d / "events.out.tfevents.complete"
        with open(comp_file, "xb") as compfile:
            for f in event_files:
                with open(f, "rb") as partfile:
                    shutil.copyfileobj(partfile, compfile)
                # os.remove(f)
        comp_files.append(str(comp_file))

def reduce_tb_events(indirs_glob,
                     outdir: Union[str, Path],
                     overwrite: bool = False,
                     reduce_ops: List[str] = ["mean", "min", "max"]):
    events_dict = load_tb_events(indirs_glob)
    reduced_events = reduce_events(events_dict, reduce_ops)
    write_tb_events(reduced_events, str(outdir), overwrite)

def reduce_tb_events_from_globs(input_globs,
                                outdir: Union[str, Path],
                                overwrite: bool = False,
                                reduce_ops: List[str] = ["mean", "min", "max"]):
    events_dict = {}
    for glob in input_globs:
        run = defaultdict(list)
        event_files = sorted(list(glob))
        for efile in event_files:
            try:
                for key, data in load_tb_events(str(efile), strict_steps=False).items():
                    run[key] = np.append(run[key], data)
            except Exception:
                pass

        for key, data in run.items():
            data = data.reshape(-1,1)
            if key not in events_dict:
                events_dict[key] = data
            else:
                events_dict[key] = np.hstack((events_dict[key], data))

    reduced_events = reduce_events(events_dict, reduce_ops)
    write_tb_events(reduced_events, str(outdir), overwrite)
    # return reduced_events

def prepare_batch(
        batch: Sequence[Union[torch.Tensor, Sequence[Tensor], Mapping[Any, Tensor]]],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False
) -> Tuple[Union[torch.Tensor, Sequence, Mapping, str, bytes], ...]:
    return tuple(map(
        lambda x: convert_tensor(x, device=device, non_blocking=non_blocking),
        batch))

class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass

tensor_or_mapping = Union[Tensor, Mapping[Any, Tensor]]
class MyTensorDataset(Dataset[Tuple[tensor_or_mapping, ...]]):
    r"""Dataset wrapping tensors and dicts of tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    tensors: Tuple[tensor_or_mapping, ...]
    sizes: Tuple[int, ...]
    T = TypeVar('T')

    def __init__(self, *tensors: Union[Tensor, Mapping[Any, Tensor]]) -> None:
        self.sizes = tuple(self._flatten(map(self._to_size, tensors)))
        assert all(self.sizes[0] == size for size in self.sizes), "Size mismatch between tensors"
        self.tensors = tensors

    @staticmethod
    def _to_size(el: tensor_or_mapping) -> Union[int, Mapping[Any, int]]:
        if isinstance(el, Tensor):
            return el.size(0)
        elif isinstance(el, Mapping):
            return {k:v.size(0) for k,v in el.items()}

    @staticmethod
    def _get_item(el: tensor_or_mapping, index) -> Union[Tensor, Mapping[Any, Tensor]]:
        if isinstance(el, Tensor):
            return el[index]
        elif isinstance(el, Mapping):
            return {k:v[index] for k,v in el.items()}

    @staticmethod
    def _flatten(dat: Iterable[Union[T, Mapping[Any, T]]]) -> Generator[T, None, None]:
        for el in dat:
            if isinstance(el, Mapping):
                for i in el.values():
                    yield i
            else:
                yield el

    def __getitem__(self, index):
        return tuple((self._get_item(el, index) for el in self.tensors))

    def __len__(self):
        return self.sizes[0]
