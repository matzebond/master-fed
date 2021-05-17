import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
# from multiprocessing.pool import Pool
import ignite
from ignite import metrics
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
import wandb
from itertools import *
import dill
import os
from pathlib import Path
from time import sleep
import logging

import functools

import FedMD
from data import load_idx_from_artifact, build_private_dls
import util

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def log_training(engine):
    batch_loss = engine.state.output
    e = engine.state.epoch
    n = engine.state.max_epochs
    i = engine.state.iteration
    print(f"Epoch {e}/{n}: {i} - batch loss: {batch_loss:.4f}")


def init_pool_process(partial_dls, combined_dl, test_dl, p_train_dl, p_test_dl):
    global private_partial_dls, private_combined_dl, private_test_dl
    private_partial_dls = dill.loads(partial_dls)
    private_combined_dl = dill.loads(combined_dl)
    private_test_dl = dill.loads(test_dl)
    global public_train_dl, public_test_dl
    public_train_dl = dill.loads(p_train_dl)
    public_test_dl = dill.loads(p_test_dl)

def self_dec(func):
    @functools.wraps(func)
    def inner(self, *args, **kwds):
        ret = func(self, *args, **kwds) 
        return self, ret
    return inner

class FedWorker:
    def __init__(self, rank, cfg):
        self.cfg = cfg

        np.random.seed()
        torch.manual_seed(np.random.randint(0, 0xffff_ffff))

        print(f"start run {self.cfg['rank']} in pid {os.getpid()}")

        self.cfg['architecture'] = FedMD.FedMD_CIFAR_hyper[self.cfg['model']]
        self.model = FedMD.FedMD_CIFAR(10+len(cfg['subclasses']),
                                       (3, 32, 32),
                                       *self.cfg['architecture'])

        self.cfg['path'] = self.cfg['path'] / str(self.cfg['rank'])

        # w_run = wandb.init(project='mp-test', entity='maschm',
        #                    group=self.cfg['group'], job_type='party',
        #                    config=self.cfg,
        #                    config_exclude_keys=['group', 'rank'],
        #                    sync_tensorboard=True)
        # wandb.watch(self.model)

        self.gstep = 0

        self.optim_state = None


    def setup(self, optimizer_state=None):
        self.model = self.model.to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.cfg['init_public_lr'])
        if optimizer_state != None:
            self.optimizer.load_state_dict(optimizer_state)
            util.optim_to(self.optimizer, device)

        self.trainer = create_supervised_trainer(
            self.model,
            self.optimizer,
            nn.CrossEntropyLoss(),
            device,
        )
        self.trainer_logit = create_supervised_trainer(
            self.model,
            self.optimizer,
            nn.MSELoss(),  # TODO use KL Loss
            device,
        )
        self.evaluator = create_supervised_evaluator(
            self.model,
            {"acc": metrics.Accuracy(),
             "loss": metrics.Loss(nn.CrossEntropyLoss())},
            device)


        self.private_dl = private_partial_dls[self.cfg['rank']]
        self.public_dls  = {"public_train": public_train_dl,
                            "public_test": public_test_dl}
        self.private_dls = {"private_train": self.private_dl,
                            "private_test": private_test_dl}

        self.writer = SummaryWriter(self.cfg['path'])


    def teardown(self, save_optimizer=False):
        if save_optimizer:
            self.optim_state = util.optim_to(self.optimizer, "cpu").state_dict()

        del self.optimizer, self.trainer, self.trainer_logit, self.evaluator
        del self.private_dl, self.public_dls, self.private_dls
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

        for name, dl in dls.items():
            title = f"{stage}/{name}" if add_stage else name
            with self.evaluator.add_event_handler(Events.COMPLETED,
                                                  self.log_metrics, title):
                self.evaluator.state.max_epochs = None
                self.evaluator.run(dl)
        if advance:
            self.gstep += 1


    def coarse_eval(self):
        self.gstep -= 1
        self.evaluate(None, "coarse", {"private_test": private_test_dl}, add_stage=True)


    @self_dec
    def init_public(self):
        print(f"party {self.cfg['rank']}: start 'init_public' stage")
        self.gstep = 0
        self.setup()

        with self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.evaluate,
                                            "init_public",
                                            self.public_dls,
                                            add_stage=True):
            self.trainer.state.max_epochs = None
            self.trainer.run(public_train_dl, self.cfg['init_public_epochs'])

        torch.save(self.model.state_dict(), f"{self.cfg['path']}/init_public.pth")
        torch.save(self.optimizer.state_dict(), f"{self.cfg['path']}/init_public_optim.pth")

        self.teardown()


    @self_dec
    def init_private(self):
        print(f"party {self.cfg['rank']}: start 'init_private' stage")
        self.model.load_state_dict(torch.load(f"{self.cfg['path']}/init_public.pth"))
        self.gstep = 0
        self.setup(torch.load(f"{self.cfg['path']}/init_public_optim.pth"))

        with self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.evaluate,
                                            "init_private", self.private_dls):
            self.trainer.state.max_epochs = None
            self.trainer.run(self.private_dl, self.cfg['init_private_epochs'])

        torch.save(self.model.state_dict(), f"{self.cfg['path']}/init_private.pth")
        torch.save(self.optimizer.state_dict(), f"{self.cfg['path']}/init_private_optim.pth")

        self.teardown()


    @self_dec
    def upper_bound(self):
        self.model.load_state_dict(torch.load(f"{self.cfg['path']}/init_public.pth"))
        self.gstep = 0
        self.setup(torch.load(f"{self.cfg['path']}/init_public_optim.pth"))

        with self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.evaluate,
                                            "upper", self.private_dls, add_stage=True):
            self.trainer.state.max_epochs = None
            epochs = self.cfg['init_private_epochs'] \
                + self.cfg['collab_rounds'] * self.cfg['private_training_epochs']
            self.trainer.run(private_combined_dl, epochs)

        self.teardown()


    @self_dec
    def lower_bound(self):
        self.model.load_state_dict(torch.load(f"{self.cfg['path']}/init_private.pth"))
        self.gstep = 0
        self.setup(torch.load(f"{self.cfg['path']}/init_private_optim.pth"))

        with self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.evaluate,
                                            "lower", self.private_dls, add_stage=True):
            self.trainer.state.max_epochs = None
            epochs = self.cfg['collab_rounds'] * self.cfg['private_training_epochs']
            self.trainer.run(self.private_dl, epochs)

        self.teardown()


    @self_dec
    def start_collab(self):
        self.model.load_state_dict(torch.load(f"{self.cfg['path']}/init_private.pth"))
        self.setup(torch.load(f"{self.cfg['path']}/init_private_optim.pth"))
        print(f"party {self.cfg['rank']}: start 'collab' stage")

        gstep = self.cfg['init_private_epochs']
        self.coarse_eval()

        self.teardown(save_optimizer=True)


    @self_dec
    def get_logits(self, alignment_data):
        self.setup()
        def logit_collect(engine, batch):
            self.model.train()
            x = batch[0].to(device)
            with torch.no_grad():
                logits = self.model(x)
            if engine.state.iteration > 1:
                engine.state.logits = torch.cat((engine.state.logits, logits))
            else:
                engine.state.logits = logits

        alignment_ds = DataLoader(TensorDataset(alignment_data),
                                  batch_size=self.cfg['logits_matching_batchsize'])
        logit_collector = Engine(logit_collect)
        logit_collector.run(alignment_ds)

        self.teardown()
        return logit_collector.state.logits.cpu()


    @self_dec
    def collab_round(self, allignment_data = None, logits = None):
        self.setup(self.optim_state)

        if allignment_data != None and logits != None:
            logit_dl = DataLoader(TensorDataset(alignment_data, logits),
                                batch_size=self.cfg['logits_matching_batchsize'])
            with self.trainer_logit.add_event_handler(Events.EPOCH_COMPLETED,
                                                      self.evaluate,
                                                      "alignment", self.private_dls):
                # trainer_logit.state.gstep = trainer.state.gstep
                self.trainer_logit.state.max_epochs = None
                self.trainer_logit.run(logit_dl, self.cfg['logits_matching_epochs'])

        with self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.evaluate,
                                            "private_training", self.private_dls):
            # trainer.state.gstep = trainer_logit.state.gstep
            self.trainer.state.max_epochs = None
            self.trainer.run(self.private_dl, self.cfg['private_training_epochs'])
        
        self.coarse_eval()
        self.teardown(save_optimizer=True)


    @self_dec
    def get_model_state(self):
        return self.model.state_dict()


    @self_dec
    def set_model_state(self, state):
        self.model.load_state_dict(state)




def main():
    cfg = {
        'group': wandb.util.generate_id(),
        'samples_per_class': 3,
        'dataset': 'CIFAR100',
        'data_variant': 'iid',
        'subclasses': [0,2,20,63,71,82],
        'parties': 10,
        'load_private_idx': True,
        'optim': 'Adam',
        'init_public_lr': 0.001,
        'init_public_epochs': 10,
        'init_public_batch_size': 64,
        'init_private_epochs': 10,
        'init_private_batch_size': 32,
        'collab_rounds': 10,
        'alignment_mode': 'public',
        'num_alignment': 2000,
        'logits_matching_epochs': 10,
        'logits_matching_batchsize': 128,
        'private_training_epochs': 1,
        # 'private_training_batchsize': 5, # TODO not supported
        'model_averaging': False,
        'stages': ['init_public', 'init_private', 'collab', 'lower', 'upper'],
    }

    # model_mapping =  list(islice(cycle(range(len(FedMD.FedMD_CIFAR_hyper))), cfg['parties']))
    model_mapping = list(repeat(4, cfg['parties']))

    wandb.init(project='mp-test', entity='maschm',
               # group=cfg['group'], job_type="master", name=cfg['group'],
               config=cfg, config_exclude_keys=['group', 'rank', 'model'],
               sync_tensorboard=True)

    import CIFAR
    private_partial_idxs = load_idx_from_artifact(
        np.array(CIFAR.private_train_data.targets),
        cfg['parties'],
        cfg['subclasses'],
        cfg['samples_per_class']
    )
    private_partial_dls, private_combined_dl, private_test_dl = build_private_dls(
        CIFAR.private_train_data,
        CIFAR.private_test_data,
        10,
        cfg['subclasses'],
        private_partial_idxs,
        cfg['init_private_batch_size']
    )
    public_train_dl = DataLoader(CIFAR.public_train_data,
                                 batch_size=cfg['init_public_batch_size'])
    public_test_dl = DataLoader(CIFAR.public_test_data,
                                batch_size=cfg['init_public_batch_size'])

    print(f"train {cfg['parties']} models on")
    subclass_names = list(map(lambda x: CIFAR.private_train_data.classes[x],
                              cfg['subclasses']))
    combined_class_names = CIFAR.public_train_data.classes + subclass_names
    print("subclasses: ", subclass_names)
    print("all classes: ", combined_class_names)

    with Pool(2, init_pool_process,
              [dill.dumps(private_partial_dls),
               dill.dumps(private_combined_dl),
               dill.dumps(private_test_dl),
               dill.dumps(public_train_dl),
               dill.dumps(public_test_dl)]) as pool:
        args = []
        for i in range(cfg['parties']):
            p_cfg = cfg.copy()
            p_cfg['rank'] = i
            p_cfg['model'] = model_mapping[i]
            p_cfg['path'] = Path(wandb.run.dir)
            args.append((i,p_cfg));

        workers = pool.starmap(FedWorker, args)

        if "init_public" in cfg['stages']:
            print(f"All parties starting with 'pre-public'")
            res = pool.map(FedWorker.init_public, workers)
            [workers, res] = list(zip(*res))

        if "init_private" in cfg['stages']:
            print(f"All parties starting with 'pre-private'")
            res = pool.map(FedWorker.init_private, workers)
            [workers, res] = list(zip(*res))

        if "collab" in cfg['stages']:
            print(f"All parties starting with 'collab'")
            res = pool.map(FedWorker.start_collab, workers)
            [workers, res] = list(zip(*res))
        else:
            return

        for n in range(cfg['collab_rounds']):
            alingment_data, avg_logits = None, None
            if cfg['alignment_mode'] != "none":
                if cfg['alignment_mode'] == "public":
                    print(f"Alignment Data: {cfg['num_alignment']} random examples from the public dataset")
                    alignment_idx = np.random.choice(len(CIFAR.public_train_data),
                                                     cfg['num_alignment'],
                                                     replace = False)
                    alignment_dl = DataLoader(Subset(CIFAR.public_train_data,
                                                     alignment_idx),
                                              batch_size=cfg['num_alignment'])
                    alignment_data, alignment_labels = next(iter(alignment_dl))
                elif cfg['alignment_mode'] == "random":
                    print(f"Alignment Data: {cfg['num_alignment']} random noise inputs")
                    alignment_data = torch.rand([cfg['num_alignment']]
                                                + list(private_train_data[0][0].shape))
                else:
                    raise NotImplementedError(f"alignment_mode '{cfg['alignment_mode']}' is unknown")

                res = pool.starmap(FedWorker.get_logits,
                                   zip(workers, repeat(alignment_data)))
                [workers, logits] = list(zip(*res))

                avg_logits = torch.zeros_like(logits[0])
                for l in logits:
                    avg_logits += l
                avg_logits /= len(logits)

            res = pool.starmap(FedWorker.collab_round,
                                zip(workers, repeat(avg_logits)))
            [workers, _] = list(zip(*res))

            if cfg['model_averaging']:
                pass

        if "upper" in cfg['stages']:
            print(f"All parties starting with 'upper'")
            res = pool.map(FedWorker.upper_bound, workers)
            [workers, _] = list(zip(*res))

        if "lower" in cfg['stages']:
            print(f"All parties starting with 'lower'")
            res = pool.map(FedWorker.lower_bound, workers)
            [workers, _] = list(zip(*res))

    wandb.finish()


if __name__ == '__main__':
    mp.set_start_method('spawn')#, force=True)
    main()
