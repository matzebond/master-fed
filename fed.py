#!/usr/bin/env python

import functools
from itertools import repeat
from collections import defaultdict
import logging
import copy
import random
import os
import sys
from pathlib import Path
import shutil
import argparse
import dill
import wandb
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
# from multiprocessing.pool import Pool
import ignite
import ignite.metrics
from ignite.engine import (Engine, Events,
                           create_supervised_trainer,
                           create_supervised_evaluator)
from sklearn.neighbors import NearestNeighbors
import thop
import ast

import models
from data import get_pub_priv, load_idx_from_artifact, build_private_dls
import util
from util import MyTensorDataset, set_seed
from nn import (KLDivSoftmaxLoss, avg_params, optim_to)


individual_cfgs = ['model_mapping', 'init_public_epochs', 'init_private_epochs', 'collab_participation', 'private_training_epochs']

def build_parser():
    def float_or_string(string):
        try:
            return float(string)
        except:
            return string

    def alignment_size_type(string):
        if string == "full":
            return "full"
        return int(string)

    def slice_parser(string):
        return slice(*map(int, string.split(':')))

    parser = argparse.ArgumentParser(
        usage='%(prog)s [path/default_config_file.py] [options]'
    )

    base = parser.add_argument_group('basic')
    base.add_argument('--parties', default=2, type=int, metavar='NUM',
                      help='number of parties participating in the collaborative training')
    base.add_argument('--collab_rounds', default=5, type=int, metavar='NUM',
                      help='number of round in the collaboration phase')
    base.add_argument('--stages', nargs='*',
                      default=['init_public', 'init_private', 'collab', 'upper', 'lower'],
                      choices=['global_init_public', 'save_global_init_public', 'load_global_init_public', 'init_public', 'save_init_public', 'load_init_public', 'init_private', 'save_init_private', 'load_init_private', 'collab', 'save_collab', 'upper', 'lower'],
                      help='list of phases that are executed (default: %(default)s)')

    # model
    model = parser.add_argument_group('model')
    model.add_argument('--model_variant', default='FedMD_CIFAR', choices=['FedMD_CIFAR', 'LLP', 'LeNet_plus_plus'],
                       help='')
    model.add_argument('--model_mapping', nargs='*', type=int,
                       help='')
    model.add_argument('--global_model_mapping', type=int,
                       help='')
    model.add_argument('--projection_head', nargs='*', type=int, metavar='LAYER_SIZE',
                       help='size of the projection head')
    model.add_argument('--global_projection_head', type=int, metavar='LAYER_SIZE',
                       help='size of the projection head')

    # data
    data = parser.add_argument_group('data')
    data.add_argument('--dataset', default='CIFAR100', choices=['CIFAR100', 'CIFAR10', 'MNIST'],
                      help='')
    data.add_argument('--public_dataset', default=None, choices=['CIFAR100', 'CIFAR10', 'MNIST', 'same'],
                      help='')
    data.add_argument('--classes', nargs='+', type=int,
                      metavar='CLASS',
                      help='subset of classes that are used for the private and collaborative training. If empty or not defined all classes of the private dataset are used.')
    data.add_argument('--classes-list', type=ast.literal_eval, dest='classes',
                      metavar='CLASSLIST',
                      help='subset of classes that are used for the private and collaborative training. If empty or not defined all classes of the private dataset are used.')

    data.add_argument('--partition_normalize', default='class',
                      choices=['class', 'party', 'moon'],
                      help='select the target against which to normalize the partition in. Eather all parties or all classes have the same amount of data')
    data.add_argument('--concentration', default=1, type=float_or_string,
                      metavar='BETA',
                      help='parameter of the dirichlet distribution used to produce a non-iid data distribution for the private data, a higher values will produce more iid distributions (a float value, "iid" or "single_class" is expected)')
    data.add_argument('--samples', type=int, metavar='SAMPLES',
                      help='if normalizing per class specifies the samples per class chosen from the training dataset. if normalizing per party specifies the samples per party chosen from the training dataset. The default is to use all data and not to sample down.')
    data.add_argument('--partition_overlap', action='store_true',
                      help='')
    data.add_argument('--no-augmentation', action='store_false', dest="augmentation",
                      help="don't use augmentation in the data preprocessing")
    data.add_argument('--no-normalization', action='store_false', dest="normalization",
                      help="don't use normalization in the data preprocessing")
    data.add_argument('--datafolder', default='data', type=str,
                      help="place of the datasets (will be downloaded if not present)")

    # training
    training = parser.add_argument_group('training')
    training.add_argument('--collab_participation', nargs='+',
                          default=[slice(0, None)], type=slice_parser,
                          help="")
    training.add_argument('--private_training_epochs', nargs='+', default=5, type=int,
                          metavar='EPOCHS',
                          help='number of training epochs on the private data per collaborative round')
    # training.add_argument('--private_training_batch_size', default=5) # TODO not supported
    training.add_argument('--optim', default='Adam', choices=['Adam', 'SGD'])
    training.add_argument('--optim_lr', default=0.0001, type=float, metavar='LR',
                          help='learning rate for any training optimizer')
    training.add_argument('--optim_weight_decay', default=0, type=float, metavar='WD',
                          help='weight decay rate for any training optimizer')
    # training.add_argument('--optim_public_lr', default=0.0001, type=float,
    #                       metavar='LR',
    #                       help='public training learning rate')

    training.add_argument('--init_public_epochs', default=0, nargs='+', type=int,
                          metavar='EPOCHS',
                          help='number of training epochs on the public data in the initial public training stage')
    training.add_argument('--global_init_public_epochs', default=None, type=int,
                          metavar='EPOCHS',
                          help='number of training epochs the global model should pretrain on the public data')
    training.add_argument('--init_public_batch_size', default=32, type=int,
                          metavar='BATCHSIZE',
                          help='size of the mini-batches in the initial public training')
    training.add_argument('--init_private_epochs', default=0, nargs='+', type=int,
                          metavar='EPOCHS',
                          help='number of training epochs on the private data in the initial private training stage')
    training.add_argument('--init_private_batch_size', default=32, type=int,
                          metavar='BATCHSIZE',
                          help='size of the mini-batches in the initial private training')
    training.add_argument('--upper_bound_epochs', default=50, type=int, metavar='EPOCHS')
    training.add_argument('--lower_bound_epochs', default=50, type=int, metavar='EPOCHS')

    # variant
    variant = parser.add_argument_group('variant')
    variant.add_argument('--variant', nargs="?",
                         choices=['fedmd', 'fedavg', 'moon', 'fedcon'],
                         help='algorithm to use for the collaborative training, fixes some of the parameters in the \'variant\' group')
    variant.add_argument('--keep_prev_model', action='store_true',
                         help='parties keep the previous model (used for MOON contrastive loss)')
    variant.add_argument('--global_model', nargs="?",
                         choices=['fix', 'averaging', 'distillation'],
                         help='build a global model by the specified method')
    variant.add_argument('--replace_local_model', action='store_true',
                         help='replace the local model of the parties by the global model')
    variant.add_argument('--send_global', action='store_true',
                         help='send the global model to the parties')
    variant.add_argument('--contrastive_loss', nargs="?", choices=['moon'],
                         help='contrastive loss for the collaborative training')
    variant.add_argument('--contrastive_loss_weight', type=int,
                         metavar='WEIGHT')
    variant.add_argument('--contrastive_loss_temperature', type=int,
                         metavar='TEMP')
    variant.add_argument('--alignment_data', nargs="?",
                         choices=['public', 'random'],
                         help='data to use for the alignment step')
    variant.add_argument('--alignment_target', nargs="?",
                         choices=['logits', 'rep', 'both'],
                         help='target to use for the alignment step')
    variant.add_argument('--alignment_label_loss', action='store_true',
                         help='use crossentropy to calculate loss on alignment targets')
    variant.add_argument('--alignment_distillation_loss', nargs="?",
                         choices=['MSE', 'L1', 'SmoothL1', 'KL'],
                         help='loss to align the alignment target on the alignment data')
    variant.add_argument('--alignment_distillation_target', nargs="?",
                         choices=['logits', 'rep', 'both'],
                         help='target to use for the distillation loss')
    variant.add_argument('--alignment_distillation_weight', default=1, type=float,
                         metavar='WEIGHT',
                         help="weight for the alignment distillation loss")
    variant.add_argument('--alignment_contrastive_loss', nargs="?",
                         choices=['contrastive', 'locality_preserving'],
                         help='contrastive loss to align the alignment target on the alignment data')
    variant.add_argument('--alignment_size', type=alignment_size_type, metavar='SIZE',
                         help='amount of instances to pick from the data for the alignment')
    variant.add_argument('--alignment_aggregate', nargs="?",
                         default='mean', choices=['mean', 'first', 'all', 'global'],
                         help='how to aggregate the alignment outputs of all parties')
    variant.add_argument('--alignment_matching_epochs', type=int,
                         metavar='EPOCHS',
                         help='number of training epoch on the alignment data per collaborative round')
    variant.add_argument('--alignment_matching_batch_size', type=int, metavar='BATCHSIZE',
                         help='size of the mini-batches in the alignment')
    variant.add_argument('--alignment_temperature', type=int, metavar='TEMP',
                         help='temperature  alignment')

    # util
    util = parser.add_argument_group('etc')
    util.add_argument('--pool_size', default=1, type=int, metavar='SIZE',
                      help='number of processes')
    util.add_argument('--seed', default=0, type=int,
                      help="the seed with which to initialize numpy, torch, cuda and random")
    util.add_argument('--resumable', action='store_true',
                      help="resume form the previous run (need to be resumable) if it chrashed")


    return parser


def init_pool_process(priv_dls, priv_test_dl,
                      combi_dl, combi_test_dl,
                      pub_train_dl, pub_test_dl,
                      gpus=torch.cuda.device_count()):
    global private_dls, private_test_dls
    private_dls = dill.loads(priv_dls)
    private_test_dls = dill.loads(priv_test_dl)
    global combined_dl, combined_test_dl
    combined_dl = dill.loads(combi_dl)
    combined_test_dl = dill.loads(combi_test_dl)
    global public_train_dl, public_test_dl
    public_train_dl = dill.loads(pub_train_dl)
    public_test_dl = dill.loads(pub_test_dl)

    global device
    if torch.cuda.is_available():
        worker = mp.current_process()
        worker_id = worker._identity[0] - 1 if hasattr(worker,'_identity') else 0
        gpu_id = worker_id % gpus
        device = torch.device('cuda', gpu_id)
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")


# A hacky decorator that makes a (class) function return
# self and the result of the function, so the class can update self
# when called multiple times in a multiprocessing.Pool
def self_dec(func):
    @functools.wraps(func)
    def inner(self, *args, **kwds):
        ret = func(self, *args, **kwds)
        return self, ret
    return inner


class FedWorker:
    def __init__(self, cfg, model):
        self.cfg = cfg
        print(f"start run {self.cfg['rank']} in pid {os.getpid()}")

        set_seed(cfg['seed'])

        os.makedirs(self.cfg['path'])
        os.makedirs(self.cfg['tmp'])
        self.model = model
        self.prev_model = None
        self.gstep = 0
        self.optim_state = None
        self.device = device

    def setup(self, optimizer_state=None, writer=True):
        self.decive = device
        self.model = self.model.to(self.device)

        if self.cfg['optim'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg['optim_lr'],
                weight_decay=self.cfg['optim_weight_decay'])
        elif self.cfg['optim'] == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg['optim_lr'],
                weight_decay=self.cfg['optim_weight_decay'])
        else:
            raise Exception('unknown optimizer ' + self.cfg['optim'])

        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
            optim_to(self.optimizer, self.device)

        self.trainer = create_supervised_trainer(
            self.model,
            self.optimizer,
            nn.CrossEntropyLoss(),
            self.device,
        )
        self.evaluator = create_supervised_evaluator(
            self.model,
            {"acc": ignite.metrics.Accuracy(),
             "loss": ignite.metrics.Loss(nn.CrossEntropyLoss())},
            self.device)


        self.private_dl = private_dls[self.cfg['rank']]
        self.private_test_dl = private_test_dls[self.cfg['rank']]
        self.public_dls  = {"public_train": public_train_dl,
                            "public_test": public_test_dl}
        self.private_dls = {"private_train": self.private_dl,
                            "private_test": self.private_test_dl,
                            "combined_test": combined_test_dl}

        self.writer = SummaryWriter(self.cfg['path']) if writer else None


    def teardown(self, save_optimizer=False):
        if save_optimizer:
            self.optim_state = optim_to(self.optimizer, "cpu").state_dict()

        del self.optimizer, self.trainer, self.evaluator
        del self.private_dl, self.private_test_dl, self.public_dls, self.private_dls
        del self.writer

        self.model = self.model.cpu()


    def finish(self):
        pass


    def log_metrics(self, engine, title):
        # gstep = trainer.state.gstep
        acc = engine.state.metrics['acc']
        loss = engine.state.metrics['loss']
        print(f"{title} [{self.gstep:4}] - acc: {acc:.3f} loss: {loss:.4f}")
        self.writer.add_scalar(f"{title}/acc", acc, self.gstep)
        self.writer.add_scalar(f"{title}/loss", loss, self.gstep)


    def evaluate(self, trainer, stage, dls, add_stage=False, advance=True):
        if trainer:
            print(f"{stage} [{trainer.state.epoch:2d}/{trainer.state.max_epochs:2d}]")

            losses = []
            try:
                total_loss, *losses = trainer.state.output
                total_loss = total_loss.item()
                losses = [l.item() for l in losses]
            except Exception:
                total_loss = trainer.state.output

            title = f"{stage}/training" if add_stage else "training"
            print(f"{title} loss:", total_loss, "- parts:", *losses)
            self.writer.add_scalar(f"{title}/loss", total_loss, self.gstep)
            for i, loss in enumerate(losses):
                self.writer.add_scalar(f"{title}/loss_{i+1}", loss, self.gstep)

        res = {}
        for name, dl in dls.items():
            title = f"{stage}/{name}" if add_stage else name
            with self.evaluator.add_event_handler(Events.COMPLETED,
                                                  self.log_metrics, title):
                self.evaluator.run(dl)
                res[f"{title}/acc"] = self.evaluator.state.metrics['acc']
                res[f"{title}/loss"] = self.evaluator.state.metrics['loss']
        if advance:
            self.gstep += 1
        if trainer:
            trainer.state.eval_res = res
        return res


    def coarse_eval(self, dls, last_trainer=None):
        self.gstep = max(0, self.gstep-1)

        if last_trainer and hasattr(last_trainer.state, "eval_res"):
            last_res = last_trainer.state.eval_res
            for key in list(last_res.keys()):
                last_res["coarse/"+key] = last_res[key]
                self.writer.add_scalar("coarse/"+key, last_res[key], self.gstep)
                del last_res[key]
            self.gstep += 1
            return last_res

        return self.evaluate(None, "coarse", dls, add_stage=True)


    @self_dec
    def init_public(self, epochs=None):
        print(f"party {self.cfg['rank']}: start 'init_public' stage")
        self.model.change_classes(self.cfg['num_public_classes'])
        self.gstep = 0
        self.setup()

        with self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.evaluate,
                                            "init_public",
                                            self.public_dls,
                                            add_stage=True):
            if epochs is None:
                epochs = self.cfg['init_public_epochs']
            self.trainer.run(public_train_dl, epochs)

        torch.save(self.model.state_dict(), f"{self.cfg['tmp']}/init_public.pth")
        torch.save(self.optimizer.state_dict(), f"{self.cfg['tmp']}/init_public_optim.pth")

        res = 0, 0
        if epochs > 0:
            res = self.evaluator.state.metrics['acc'], self.evaluator.state.metrics['loss']
        self.teardown()
        return res


    @self_dec
    def init_private(self, epochs=None):
        print(f"party {self.cfg['rank']}: start 'init_private' stage")
        self.model.load_state_dict(torch.load(f"{self.cfg['tmp']}/init_public.pth"))
        self.model.change_classes(self.cfg['num_private_classes'])
        self.gstep = 0
        self.setup(torch.load(f"{self.cfg['tmp']}/init_public_optim.pth"))

        with self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.evaluate,
                                            "init_private", self.private_dls):
            if epochs is None:
                epochs = self.cfg['init_private_epochs']
            self.trainer.run(self.private_dl, epochs)

        if epochs > 0:
            res = self.evaluator.state.metrics['acc'], self.evaluator.state.metrics['loss']
            # self.trainer.state.eval_res
            torch.save(self.model.state_dict(), f"{self.cfg['tmp']}/init_private.pth")
            torch.save(self.optimizer.state_dict(), f"{self.cfg['tmp']}/init_private_optim.pth")
        else:
            res = 0, 0
            (self.cfg['tmp'] / "init_private.pth").symlink_to("init_public.pth")
            (self.cfg['tmp'] / "init_private_optim.pth").symlink_to("init_public_optim.pth")
        self.teardown()
        return res


    @self_dec
    def upper_bound(self, epochs=None):
        self.model.change_classes(self.cfg['num_public_classes'])
        self.model.load_state_dict(torch.load(f"{self.cfg['tmp']}/init_public.pth"))
        self.model.change_classes(self.cfg['num_private_classes'])
        self.gstep = 0
        self.setup(torch.load(f"{self.cfg['tmp']}/init_public_optim.pth"))

        with self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.evaluate,
                                            "upper", self.private_dls, add_stage=True):
            if epochs is None:
                if 'upper_bound_epochs' in self.cfg:
                    epochs = self.cfg['upper_bound_epochs']
                else:
                    epochs = self.cfg['init_private_epochs'] \
                        + self.cfg['collab_rounds'] * self.cfg['private_training_epochs']
            self.trainer.run(combined_dl, epochs)

        res = self.evaluator.state.metrics['acc'], self.evaluator.state.metrics['loss']
        self.teardown()
        return res


    @self_dec
    def lower_bound(self, epochs=None):
        self.model.load_state_dict(torch.load(f"{self.cfg['tmp']}/init_private.pth"))
        self.gstep = 0
        self.setup(torch.load(f"{self.cfg['tmp']}/init_private_optim.pth"))

        with self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.evaluate,
                                            "lower", self.private_dls, add_stage=True):
            if epochs is None:
                if 'lower_bound_epochs' in self.cfg:
                    epochs = self.cfg['lower_bound_epochs']
                else:
                    epochs = self.cfg['collab_rounds'] * self.cfg['private_training_epochs']
            self.trainer.run(self.private_dl, epochs)

        res = self.evaluator.state.metrics['acc'], self.evaluator.state.metrics['loss']
        self.teardown()
        return res


    @self_dec
    def start_collab(self):
        self.model.load_state_dict(torch.load(f"{self.cfg['tmp']}/init_private.pth"))
        self.setup(torch.load(f"{self.cfg['tmp']}/init_private_optim.pth"))
        print(f"party {self.cfg['rank']}: start 'collab' stage")

        # self.gstep = self.cfg['init_private_epochs']
        res = self.coarse_eval(self.private_dls)

        self.teardown(save_optimizer=True)
        return res


    def get_alignment(self, alignment_data):
        self.setup(writer=False)
        def collect_alignment(engine, batch):
            self.model.train()
            x = batch[0].to(self.device)
            with torch.no_grad():
                logits, rep = self.model(x, output='both')
                if self.cfg['alignment_target'] == 'rep' \
                   or self.cfg['alignment_target'] == 'both':
                    if self.cfg['alignment_distillation_loss'] == 'KL':
                        rep = F.softmax(rep, dim=-1)
                    engine.state.rep = torch.cat((engine.state.rep, rep.cpu()), dim=0)
                if self.cfg['alignment_target'] == 'logits' \
                   or self.cfg['alignment_target'] == 'both':
                    logits = logits / self.cfg['alignment_temperature']
                    if self.cfg['alignment_distillation_loss'] == 'KL':
                        logits = F.softmax(logits, dim=-1)
                    engine.state.logits = torch.cat((engine.state.logits, logits.cpu()), dim=0)

        alignment_ds = DataLoader(TensorDataset(alignment_data),
                                  batch_size=self.cfg['alignment_matching_batch_size'])
        alignment_collector = Engine(collect_alignment)
        alignment_collector.state.logits = torch.tensor([])
        alignment_collector.state.rep = torch.tensor([])
        alignment_collector.run(alignment_ds)

        self.teardown()

        if self.cfg['alignment_target'] == 'logits':
            return {'logits': alignment_collector.state.logits}
        elif self.cfg['alignment_target'] == 'rep':
            return {'rep': alignment_collector.state.rep}
        elif self.cfg['alignment_target'] == 'both':
            return {'logits': alignment_collector.state.logits,
                    'rep': alignment_collector.state.rep}
        else:
            raise NotImplementedError(f"alignment_target '{self.cfg['alignment_target']}' is unknown")


    def train_alignment(self, _, batch):
        self.model.train()
        self.optimizer.zero_grad()
        x, y, targets = util.prepare_batch(batch, device=self.device)
        local_logits, local_rep = self.model(x, output='both')
        losses = []

        if self.cfg['alignment_contrastive_loss'] and \
           self.cfg['alignment_contrastive_loss'] == 'contrastive':
            num = x.size(0)
            cos_dists = torch.tensor([], device=self.device)
            for i in range(num):
                tmp = torch.tile(local_rep[i], (num, 1))
                cos = F.cosine_similarity(tmp, targets['rep']).reshape(1, -1)
                cos_dists = torch.cat((cos_dists, cos), dim=0)
                # print(self.cfg['rank'], cos)

            cos_dists /= self.cfg['contrastive_loss_temperature']

            labels = torch.tensor(range(num), device=self.device, dtype=torch.long)
            loss_contrastive = F.cross_entropy(cos_dists, labels)
            losses.append(self.cfg['contrastive_loss_weight'] * loss_contrastive)

        if self.cfg['alignment_contrastive_loss'] and \
           self.cfg['alignment_contrastive_loss'] == 'locality_preserving':
            # norm2 = lambda u, v: ((u-v)**2).sum()
            # k = self.cfg['locality_preserving_k'] + 1
            nbrs = NearestNeighbors(n_neighbors=self.cfg['locality_preserving_k'] + 1,
                                    algorithm='ball_tree')
                                    # metric="pyfunc",
                                    # metric_params={"func": norm2})
            targs = targets['rep'].cpu()
            nbrs = nbrs.fit(targs)
            alpha = nbrs.kneighbors_graph(targs, mode='distance')
            # g = g.eliminate_zeros()
            alpha.data = np.exp(-alpha.data)
            alphaT = torch.tensor(alpha.toarray(), device=self.device)

            # dists = scidist.squareform(scidist.cdist(local_rep, norm2))

            dists = torch.cdist(local_rep, local_rep)
            loss_locality = torch.sum(torch.mul(dists, alphaT))
            losses.append(self.cfg['contrastive_loss_weight'] * loss_locality)

        if self.cfg['alignment_distillation_loss']:
            if self.cfg['alignment_distillation_target'] == 'logits' \
                or self.cfg['alignment_distillation_target'] == 'both':
                local_logits = local_logits / self.cfg['alignment_temperature']
                distill_loss = self.alignment_loss_fn(local_logits, targets['logits'])
                losses.append(self.cfg['alignment_distillation_weight'] * distill_loss)
            if self.cfg['alignment_distillation_target'] == 'rep' \
                or self.cfg['alignment_distillation_target'] == 'both':
                local_rep = local_rep / self.cfg['alignment_temperature']
                distill_loss = self.alignment_loss_fn(local_logits, targets['rep'])
                losses.append(self.cfg['alignment_distillation_weight'] * distill_loss)

        if self.cfg['alignment_label_loss']:
            label_loss = F.cross_entropy(local_logits, y)
            losses.append(label_loss)

        loss = sum(losses)
        loss.backward()

        self.optimizer.step()
        return loss.detach(), *(l.detach() for l in losses)


    @self_dec
    def collab_round(self, alignment_data=None, alignment_labels=None,
                     alignment_targets=None, global_model=None):
        self.setup(self.optim_state)

        if alignment_data != None and alignment_targets != None:
            self.alignment_loss_fn = None
            if self.cfg['alignment_distillation_loss'] == "MSE":
                self.alignment_loss_fn = nn.MSELoss()
            if self.cfg['alignment_distillation_loss'] == "L1":
                self.alignment_loss_fn = nn.L1Loss()
            if self.cfg['alignment_distillation_loss'] == "SmoothL1":
                self.alignment_loss_fn = nn.SmoothL1Loss()
            if self.cfg['alignment_distillation_loss'] == "KL":
                self.alignment_loss_fn = KLDivSoftmaxLoss()
            alignment_ds = MyTensorDataset(alignment_data, alignment_labels,
                                           alignment_targets)
            alignment_dl = DataLoader(alignment_ds, shuffle=True,
                                      batch_size=self.cfg['alignment_matching_batch_size'])
            alignment_tr = Engine(self.train_alignment)
            with alignment_tr.add_event_handler(Events.EPOCH_COMPLETED,
                                                        self.evaluate,
                                                        "alignment", self.private_dls):
                alignment_tr.run(alignment_dl, self.cfg['alignment_matching_epochs'])
            del self.alignment_loss_fn

        if global_model:
            self.global_model = global_model.to(self.device)
            self.global_model.eval()
        if self.prev_model:
            self.prev_model = self.prev_model.to(self.device)
            self.prev_model.eval()

        collab_tr = Engine(self.train_collab)

        with collab_tr.add_event_handler(Events.EPOCH_COMPLETED, self.evaluate,
                                            "private_training", self.private_dls):
            collab_tr.run(self.private_dl, self.cfg['private_training_epochs'])
        
        if self.cfg['private_training_epochs'] > 0:
            res = self.coarse_eval(self.private_dls, collab_tr)
        else:
            res = self.coarse_eval(self.private_dls, alignment_tr)

        self.teardown(save_optimizer=True)
        if self.cfg['keep_prev_model']:
            del self.prev_model
            self.prev_model = copy.deepcopy(self.model)
        if global_model:
            del self.global_model
        return res


    def train_collab(self, _, batch):
        self.model.train()
        self.optimizer.zero_grad()
        x, y = util.prepare_batch(batch, device=self.device)
        local_logits, local_rep = self.model(x, output='both')
        losses = []

        loss_target = F.cross_entropy(local_logits, y)
        losses.append(loss_target)

        if self.cfg['contrastive_loss'] == 'moon' and self.prev_model:
            global_rep = self.global_model(x, output='rep_only')
            prev_rep = self.prev_model(x, output='rep_only')

            pos = F.cosine_similarity(local_rep, global_rep, dim=-1).reshape(-1,1)
            neg = F.cosine_similarity(local_rep, prev_rep, dim=-1).reshape(-1,1)

            logits = torch.cat((pos, neg), dim=1)
            logits /= self.cfg['contrastive_loss_temperature']

            # first "class" sim(global_rep) is the ground truth
            labels = torch.zeros(x.size(0), device=self.device).long()
            loss_moon = F.cross_entropy(logits, labels)
            losses.append(self.cfg['contrastive_loss_weight'] * loss_moon)

        loss = sum(losses)
        loss.backward()
        self.optimizer.step()
        return loss.detach(), *(l.detach() for l in losses)


class FedGlobalWorker(FedWorker):
    def __init__(self, cfg, model):
        cfg['rank'] = -1
        super(FedGlobalWorker, self).__init__(cfg, model)

    def update_averaging(self, *models: nn.Module):
        global_weights = copy.deepcopy(models[0].state_dict())
        for model in models[1:]:
            model_weights = model.state_dict()
            for key in model_weights:
                global_weights[key] += model_weights[key]

        for key in global_weights:
            global_weights[key] = global_weights[key] / len(models)

        self.model.load_state_dict(global_weights)

        combined_test_dl
        public_test_dl


def fed_main(cfg):
    # wandb.tensorboard.patch(root_logdir="wandb/latest-run/files")
    wandb.init(project='master-fed', entity='maschm',
               config=cfg, sync_tensorboard=True,
               resume=cfg['resumable'])
    print(cfg['resumable'], wandb.run.resumed)

    # wandb.save("./*/*event*", policy = 'end')
    cfg['path'] = Path(wandb.run.dir)
    cfg['tmp'] = cfg['path'] / '..' / 'tmp'

    if cfg['resumable']:
        cfg['tmp'] = Path("./wandb/tmp/")
        if not wandb.run.resumed:
            shutil.rmtree(cfg['tmp'], ignore_errors=True)

    # wandb.tensorboard.patch(root_logdir=cfg['path'])

    public_train_data, public_test_data, private_train_data, private_test_data = \
        get_pub_priv(cfg['dataset'],
                     public=cfg['public_dataset'],
                     root=cfg['datafolder'],
                     augment=cfg['augmentation'],
                     normalize=cfg['normalization'])

    if cfg['classes'] is None or len(cfg['classes']) == 0:
        cfg['classes'] = list(range(len(private_train_data.classes)))
    else:
        for c in cfg['classes']:
            if c >= len(private_train_data.classes) or c < 0:
                raise Exception("--classes out of range")

    cfg['num_public_classes'] = len(public_train_data.classes)
    cfg['num_private_classes'] = len(cfg['classes'])

    private_idxs, private_test_idxs = load_idx_from_artifact(
        cfg,
        np.array(private_train_data.targets),
        np.array(private_test_data.targets),
    )

    global private_dls, private_test_dls
    global combined_dl, combined_test_dl
    private_dls, private_test_dls, combined_dl, combined_test_dl = build_private_dls(
        private_train_data,
        private_test_data,
        private_idxs, private_test_idxs,
        cfg['classes'],
        cfg['init_private_batch_size']
    )
    global public_train_dl, public_test_dl
    if cfg['public_dataset'] == 'same':
        public_train_dl = combined_dl
        public_test_dl = combined_test_dl
    else:
        public_train_dl = DataLoader(public_train_data,
                                     batch_size=cfg['init_public_batch_size'])
        public_test_dl = DataLoader(public_test_data,
                                    batch_size=cfg['init_public_batch_size'])

    input_shape = private_train_data[0][0].shape
    if len(input_shape) < 3:
        input_shape = (1, *input_shape)

    print(f"train {cfg['parties']} models on")
    class_names = [private_train_data.classes[x] for x in cfg['classes']]
    print("public classes:", public_train_data.classes)
    print("private classes: ", class_names)
    print("private_dls length:", *(len(d.dataset) for d in private_dls))
    print("combined_dl length:", len(combined_dl.dataset))
    print("public_dl length:", len(public_train_dl.dataset))


    # to mitigate divergence due to loading the indices
    set_seed(cfg['seed'])

    workers = []
    global_worker = None
    stages_todo = []
    start_round = 0

    # init models
    if not wandb.run.resumed:
        stages_todo = cfg['stages']
        for i in range(cfg['parties']):
            w_cfg = cfg.copy()
            w_cfg['rank'] = i
            w_cfg['path'] = cfg['path'] / str(w_cfg['rank'])
            w_cfg['tmp'] = cfg['tmp'] / str(w_cfg['rank'])
            for c in individual_cfgs:
                w_cfg[c] = cfg[c][i]
            model = getattr(models, w_cfg['model_variant'])
            w_cfg['architecture'] = model.hyper[w_cfg['model_mapping']]
            model = model(*w_cfg['architecture'],
                            projection = w_cfg['projection_head'],
                            n_classes = w_cfg['num_private_classes'],
                            input_size = input_shape)
            workers.append(FedWorker(w_cfg, model))

        model = copy.deepcopy(workers[0].model)
        test_input = public_train_data[0][0].reshape((1,*input_shape))
        macs, params = thop.profile(model, inputs=(test_input, ))
        print("worker macs&params: ", *thop.clever_format([macs, params], "%.3f"))
        wandb.config.update({"model": {"macs": macs, "params": params}})

        if cfg['global_model']:
            w_cfg = cfg.copy()
            w_cfg['rank'] = -1
            w_cfg['path'] = cfg['path'] / str(w_cfg['rank'])
            w_cfg['tmp'] = cfg['tmp'] / str(w_cfg['rank'])
            model = getattr(models, w_cfg['model_variant'])
            w_cfg['model_mapping'] = cfg['global_model_mapping']
            w_cfg['global_architecture'] = model.hyper[w_cfg['model_mapping']]

            model = model(*w_cfg['global_architecture'],
                            projection = w_cfg['global_projection_head'],
                            n_classes = w_cfg['num_private_classes'],
                            input_size = public_train_data[0][0].shape)
            global_worker = FedGlobalWorker(w_cfg, model)
            macs, params = thop.profile(model, inputs=(test_input, ), verbose=False)
            print("global macs&params: ", *thop.clever_format([macs, params], "%.3f"))

    if wandb.run.resumed: # else case of init models
        workers = torch.load(cfg['tmp'] / "workers.pt")
        if cfg['global_model']:
            global_worker = torch.load(cfg['tmp'] / "global.pt")
        state = torch.load(cfg['tmp'] / "state.pt")
        start_round = state['round'] + 1
        stages_todo = state['stages_todo']

    if "global_init_public" in stages_todo:
        _, (acc, loss) = global_worker.init_public(cfg['global_init_public_epochs'])
        wandb.run.summary["global_init_public/acc"] = acc
        wandb.run.summary["global_init_public/loss"] = loss
        if "save_global_init_public" in stages_todo:
            util.save_models_to_artifact(cfg, [global_worker],
                                         "global_init_public",
                                         {"acc": acc, "loss": loss},
                                         filename="init_public")
        global_worker.init_private(0)
    elif "load_global_init_public" in stages_todo:
        global_worker.model.change_classes(cfg['num_public_classes'])
        util.load_models_from_artifact(cfg, [global_worker],
                                       "global_init_public",
                                       filename="init_public")
        global_worker.init_private(0)

    pool = mp.Pool(cfg['pool_size'], init_pool_process,
                   [dill.dumps(private_dls),
                    dill.dumps(private_test_dls),
                    dill.dumps(combined_dl),
                    dill.dumps(combined_test_dl),
                    dill.dumps(public_train_dl),
                    dill.dumps(public_test_dl)])

    if "init_public" in stages_todo:
        print("All parties starting with 'init_public'")
        res = pool.map(FedWorker.init_public, workers)
        [workers, res] = list(zip(*res))
        [acc, loss] = list(zip(*res))
        acc, loss = np.average(acc), np.average(loss)
        wandb.run.summary["init_public/acc"] = acc
        wandb.run.summary["init_public/loss"] = loss
        if "save_init_public" in stages_todo:
            util.save_models_to_artifact(cfg, workers, "init_public",
                                        {"acc": acc, "loss": loss})
    elif "load_init_public" in stages_todo:
        for w in workers:
            w.model.change_classes(cfg['num_public_classes'])
        util.load_models_from_artifact(cfg, workers, "init_public")

    if "init_private" in stages_todo:
        print("All parties starting with 'init_private'")
        res = pool.map(FedWorker.init_private, workers)
        [workers, res] = list(zip(*res))
        [acc, loss] = list(zip(*res))
        acc, loss = np.average(acc), np.average(loss)
        wandb.run.summary["init_private/acc"] = acc
        wandb.run.summary["init_private/loss"] = loss
        if "save_init_private" in stages_todo:
            util.save_models_to_artifact(cfg, workers, "init_private",
                                        {"acc": acc, "loss": loss})
    elif "load_init_private" in stages_todo:
        for w in workers:
            w.model.change_classes(cfg['num_private_classes'])
        util.load_models_from_artifact(cfg, workers, "init_private")

    if "collab" in stages_todo:
        print("All parties starting with 'collab'")
        global_worker.start_collab()
        res = pool.map(FedWorker.start_collab, workers)
        [workers, res] = list(zip(*res))

        metrics = defaultdict(float)
        for d in res:
            for k in d:
                local = k.replace("coarse", "local")
                metrics[local] += d[k]
        for k in metrics:
            metrics[k] /= len(res)

        wandb.log(metrics)

        for n in range(start_round, cfg['collab_rounds']):
            print(f"All parties starting with collab round [{n+1}/{cfg['collab_rounds']}]")
            alignment_data, alignment_labels, agg_alignment_targets = None, None, {}
            if cfg['alignment_data']:
                if cfg['alignment_data'] == "public":
                    if cfg['alignment_size'] == "full" and not cfg['samples'] is None:
                        print(f"Alignment Data: all {len(public_train_dl.dataset)} examples from the public dataset")
                        alignment_dl = DataLoader(
                            public_train_dl.dataset,
                            batch_size=len(public_train_dl.dataset))
                        alignment_data, alignment_labels = next(iter(alignment_dl))
                    elif cfg['alignment_size'] == "full":
                        print(f"Alignment Data: all {len(public_train_data)} examples from the public dataset")
                        alignment_data, alignment_labels = list(zip(*public_train_data))
                        alignment_data = torch.stack(alignment_data)
                        alignment_labels = torch.tensor(alignment_labels)
                    else:
                        print(f"Alignment Data: {cfg['alignment_size']} random examples from the public dataset")
                        idxs = np.random.choice(len(public_train_data),
                                                cfg['alignment_size'],
                                                replace = False)
                        alignment_dl = DataLoader(Subset(public_train_data, idxs),
                                                  batch_size=cfg['alignment_size'])
                        alignment_data, alignment_labels = next(iter(alignment_dl))
                elif cfg['alignment_data'] == "random":
                    print(f"Alignment Data: {cfg['alignment_size']} random noise inputs")
                    alignment_data = torch.rand([cfg['alignment_size']]
                                                + list(private_train_data[0][0].shape))
                else:
                    raise NotImplementedError(f"alignment_data '{cfg['alignment_data']}' is unknown")

                if cfg['alignment_aggregate'] == "global":
                    agg_alignment_targets = global_worker.get_alignment(alignment_data)
                else:
                    res = pool.starmap(FedWorker.get_alignment,
                                       zip(workers, repeat(alignment_data)))
                    assert len(res) > 0

                    for key in res[0]:
                        tmp = tuple(r[key] for r in res)
                        agg_alignment_targets[key] = torch.zeros_like(tmp[0])
                        if cfg['alignment_aggregate'] == "mean":
                            for t in tmp:
                                agg_alignment_targets[key] += t
                            agg_alignment_targets[key] /= len(tmp)
                        elif cfg['alignment_aggregate'] == "first":
                            agg_alignment_targets[key] = tmp[0]
                        elif cfg['alignment_aggregate'] == "all":
                            agg_alignment_targets[key] = tmp

            res = pool.starmap(FedWorker.collab_round,
                               zip(workers,
                                   repeat(alignment_data),
                                   repeat(alignment_labels),
                                   repeat(agg_alignment_targets),
                                   repeat(global_worker.model \
                                          if cfg['send_global'] else None)))
            [workers, res] = list(zip(*res))
            metrics = defaultdict(float)
            for d in res:
                for k in d:
                    local = k.replace("coarse", "local")
                    metrics[local] += d[k]
            for k in metrics:
                metrics[k] /= len(res)

            # update global model
            if cfg['global_model'] == 'averaging':
                global_worker.update_averaging(*(w.model for w in workers))
                print("model parameters averaged")
            elif cfg['global_model'] == 'distillation':
                # TODO
                pass

            # eval global model
            if cfg['global_model'] and cfg['global_model'] != 'fix':
                evaluator = create_supervised_evaluator(
                    global_worker.model.to(device),
                    {"acc": ignite.metrics.Accuracy(),
                     "loss": ignite.metrics.Loss(nn.CrossEntropyLoss())},
                    device)
                evaluator.run(combined_test_dl)
                metrics.update({"global/combined_test/acc": evaluator.state.metrics['acc'],
                                "global/combined_test/loss": evaluator.state.metrics['loss']})
                global_worker.model.cpu()
            if cfg['global_model'] and cfg['global_model'] == 'fix':
                metrics.update({"global/combined_test/acc": wandb.run.summary["global_init_public/acc"],
                                "global/combined_test/loss": wandb.run.summary["global_init_public/loss"]})
            elif not cfg['global_model']:
                metrics.update({"global/combined_test/acc": metrics['local/combined_test/acc'],
                                "global/combined_test/loss": metrics['local/combined_test/loss']})
            wandb.log(metrics)

            # TODO should this go to the start of the loop?
            if cfg['replace_local_model']:
                for w in workers:
                    w.model.load_state_dict(global_worker.model.state_dict())
                print("local models replaced")

            if cfg['resumable']:
                state = {'round': n, 'stages_todo': stages_todo}
                torch.save(state, cfg['tmp'] / "state.pt")
                torch.save(workers, cfg['tmp'] / "workers.pt")
                if cfg['global_model']:
                    torch.save(global_worker, cfg['tmp'] / "global.pt")
                print("saved resuable state for round", n)

    if "save_collab" in stages_todo:
        for w in workers:
            torch.save(w.model.state_dict(), f"{w.cfg['tmp']}/final.pth")
            torch.save(None, f"{w.cfg['tmp']}/final_optim.pth")
        util.save_models_to_artifact(cfg, workers, "final",
                                     {"acc": metrics['global/combined_test/acc'],
                                      "loss": metrics['global/combined_test/loss']})

    if "upper" in stages_todo:
        print("All parties starting with 'upper'")
        res = pool.map(FedWorker.upper_bound, workers)
        [workers, res] = list(zip(*res))
        [acc, loss] = list(zip(*res))
        wandb.run.summary["upper/acc"] = np.average(acc)
        wandb.run.summary["upper/loss"] = np.average(loss)

    if "lower" in stages_todo:
        print("All parties starting with 'lower'")
        res = pool.map(FedWorker.lower_bound, workers)
        [workers, res] = list(zip(*res))
        [acc, loss] = list(zip(*res))
        wandb.run.summary["lower/acc"] = np.average(acc)
        wandb.run.summary["lower/loss"] = np.average(loss)

    pool.close()
    pool.terminate()


    # globs = (w.cfg['path'].glob("events.out.tfevents*") for w in workers)
    # util.reduce_tb_events_from_globs(globs,
    #                                  cfg['path'] / "all",
    #                                  # reduce_ops = ["mean", "min", "max"])
    #                                  reduce_ops = ["mean"])

    wandb.finish()


if __name__ == '__main__':
    config_file = 'config_base.py'
    if len(sys.argv) >= 2 and Path(sys.argv[1]).is_file():
        config_file = sys.argv.pop(1)

    global cfg
    with open(config_file) as f:
        exec(f.read())

    parser = build_parser()
    parser.set_defaults(**cfg)
    args = parser.parse_args()
    cfg = vars(args)

    # fill parameters for all parties
    for c in individual_cfgs:
        if isinstance(cfg[c], int):
            cfg[c] = [cfg[c]] * cfg['parties']
        elif len(cfg[c]) < cfg['parties']:
            missing = cfg['parties'] - len(cfg[c])
            cfg[c] += [cfg[c][-1]] * missing


    if cfg['variant'] == 'fedmd':
        cfg['global_model'] = None
        cfg['replace_local_model'] = False
        cfg['keep_prev_model'] = False
        cfg['send_global'] = False
        cfg['alignment_data'] = 'public'
        cfg['alignment_target'] = 'logits'
        cfg['alignment_distillation_target'] = 'logits'
    elif cfg['variant'] == 'fedavg':
        cfg['global_model'] = 'averaging'
        cfg['replace_local_model'] = True
        cfg['keep_prev_model'] = False
        cfg['send_global'] = False
        cfg['alignment_data'] = None
    elif cfg['variant'] == 'moon':
        cfg['global_model'] = 'averaging'
        cfg['replace_local_model'] = True
        cfg['keep_prev_model'] = True
        cfg['send_global'] = True
        cfg['alignment_data'] = None
        cfg['contrastive_loss'] = 'moon'
    elif cfg['variant'] == 'fedcon':
        cfg['global_model'] = None
        cfg['replace_local_model'] = False
        cfg['keep_prev_model'] = False
        cfg['send_global'] = False
        cfg['alignment_data'] = 'public'
        cfg['alignment_target'] = 'both'

    print(cfg)

    if cfg['alignment_target'] != 'both' and (cfg['alignment_distillation_target'] and cfg['alignment_target'] != cfg['alignment_distillation_target']):
        raise Exception('alignment_target and alignment_distillation_target are incompatible')

    if cfg['seed'] is None:
        cfg['seed'] = np.random.randint(0, 0xffff_ffff)
    set_seed(cfg['seed'])

    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # logger = mp.log_to_stderr()
    # logger.setLevel(multiprocessing.SUBDEBUG)
    if cfg['pool_size'] == 1:
        import multiprocessing.dummy as mp
        print("Using dummy for multiprocessing")
    else:
        mp.set_start_method('spawn')  #, force=True)

    # from torch.profiler import profile, record_function, ProfilerActivity
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #     fed_main(cfg)
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    fed_main(cfg)
