import functools
import os
import logging
from itertools import repeat
from pathlib import Path
import copy

import dill
import wandb
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
# from multiprocessing.pool import Pool
import ignite
from ignite import metrics
from ignite.engine import (Engine, Events,
                           create_supervised_trainer,
                           create_supervised_evaluator,
                           _prepare_batch)

import FedMD
from data import load_idx_from_artifact, build_private_dls
import util
from nn import (KLDivSoftmaxLoss, avg_params, optim_to)


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
    def __init__(self, cfg, model):
        self.cfg = cfg
        print(f"start run {self.cfg['rank']} in pid {os.getpid()}")

        np.random.seed()
        torch.manual_seed(np.random.randint(0, 0xffff_ffff))

        os.makedirs(self.cfg['path'])
        self.model = model
        self.prev_model = None
        self.gstep = 0
        self.optim_state = None

    def setup(self, optimizer_state=None, writer=True):
        self.model = self.model.to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.cfg['init_public_lr'])
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
            optim_to(self.optimizer, device)

        self.trainer = create_supervised_trainer(
            self.model,
            self.optimizer,
            nn.CrossEntropyLoss(),
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

        self.writer = SummaryWriter(self.cfg['path']) if writer else None


    def teardown(self, save_optimizer=False):
        if save_optimizer:
            self.optim_state = optim_to(self.optimizer, "cpu").state_dict()

        del self.optimizer, self.trainer, self.evaluator
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
                self.evaluator.run(dl)
        if advance:
            self.gstep += 1

        return self.evaluator.state.metrics['acc'], self.evaluator.state.metrics['loss']


    def coarse_eval(self, dls):
        self.gstep = max(0, self.gstep-1)
        return self.evaluate(None, "coarse", dls, add_stage=True)


    @self_dec
    def init_public(self):
        print(f"party {self.cfg['rank']}: start 'init_public' stage")
        self.gstep = 0
        self.setup()

        with self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.evaluate,
                                            "init_public",
                                            self.public_dls,
                                            add_stage=True):
            self.trainer.run(public_train_dl, self.cfg['init_public_epochs'])

        torch.save(self.model.state_dict(), f"{self.cfg['path']}/init_public.pth")
        torch.save(self.optimizer.state_dict(), f"{self.cfg['path']}/init_public_optim.pth")

        res = 0, 0
        if self.cfg['init_public_epochs'] > 0:
            res = self.evaluator.state.metrics['acc'], self.evaluator.state.metrics['loss']
        self.teardown()
        return res


    @self_dec
    def init_private(self):
        print(f"party {self.cfg['rank']}: start 'init_private' stage")
        self.model.load_state_dict(torch.load(f"{self.cfg['path']}/init_public.pth"))
        self.gstep = 0
        self.setup(torch.load(f"{self.cfg['path']}/init_public_optim.pth"))

        with self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.evaluate,
                                            "init_private", self.private_dls):
            self.trainer.run(self.private_dl, self.cfg['init_private_epochs'])

        torch.save(self.model.state_dict(), f"{self.cfg['path']}/init_private.pth")
        torch.save(self.optimizer.state_dict(), f"{self.cfg['path']}/init_private_optim.pth")

        res = 0, 0
        if self.cfg['init_public_epochs'] > 0:
            res = self.evaluator.state.metrics['acc'], self.evaluator.state.metrics['loss']
        self.teardown()
        return res


    @self_dec
    def upper_bound(self):
        self.model.load_state_dict(torch.load(f"{self.cfg['path']}/init_public.pth"))
        self.gstep = 0
        self.setup(torch.load(f"{self.cfg['path']}/init_public_optim.pth"))

        with self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.evaluate,
                                            "upper", self.private_dls, add_stage=True):
            if 'upper_bound_epochs' in self.cfg:
                epochs = self.cfg['upper_bound_epochs']
            else:
                epochs = self.cfg['init_private_epochs'] \
                    + self.cfg['collab_rounds'] * self.cfg['private_training_epochs']
            self.trainer.run(private_combined_dl, epochs)

        res = self.evaluator.state.metrics['acc'], self.evaluator.state.metrics['loss']
        self.teardown()
        return res


    @self_dec
    def lower_bound(self):
        self.model.load_state_dict(torch.load(f"{self.cfg['path']}/init_private.pth"))
        self.gstep = 0
        self.setup(torch.load(f"{self.cfg['path']}/init_private_optim.pth"))

        with self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.evaluate,
                                            "lower", self.private_dls, add_stage=True):
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
        self.model.load_state_dict(torch.load(f"{self.cfg['path']}/init_private.pth"))
        self.setup(torch.load(f"{self.cfg['path']}/init_private_optim.pth"))
        print(f"party {self.cfg['rank']}: start 'collab' stage")

        gstep = self.cfg['init_private_epochs']
        res = self.coarse_eval(self.private_dls)

        self.teardown(save_optimizer=True)
        return res


    def get_logits(self, alignment_data):
        self.setup(writer=False)
        def logit_collect(engine, batch):
            self.model.train()
            x = batch[0].to(device)
            with torch.no_grad():
                logits = self.model(x) / self.cfg['logits_temperature']
            engine.state.logits = torch.cat((engine.state.logits, logits.cpu()))

        alignment_ds = DataLoader(TensorDataset(alignment_data),
                                  batch_size=self.cfg['logits_matching_batchsize'])
        logit_collector = Engine(logit_collect)
        logit_collector.state.logits = torch.tensor([])
        logit_collector.run(alignment_ds)

        self.teardown()
        return logit_collector.state.logits

    @self_dec
    def collab_round(self, alignment_data = None, logits = None, global_model = None):
        self.setup(self.optim_state)

        if alignment_data != None and logits != None:
            logit_loss_fn = nn.L1Loss() # KLDivSoftmaxLoss(), nn.MSELoss()
            def train_logit(engine, batch):
                self.model.train()
                self.optimizer.zero_grad()
                x, y = _prepare_batch(batch, device=device)
                y_pred = self.model(x) / self.cfg['logits_temperature']
                loss = logit_loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()
                return loss
            trainer_logit = Engine(train_logit)
            logit_dl = DataLoader(TensorDataset(alignment_data, logits),
                                  batch_size=self.cfg['logits_matching_batchsize'])
            with trainer_logit.add_event_handler(Events.EPOCH_COMPLETED, self.evaluate,
                                                 "alignment", self.private_dls):
                trainer_logit.run(logit_dl, self.cfg['logits_matching_epochs'])


        if global_model:
            global_model = global_model.to(device)
            global_model.eval()
        if self.prev_model:
            self.prev_model = self.prev_model.to(device)
            self.prev_model.eval()

        def train_collab(engine, batch):
            self.model.train()
            self.optimizer.zero_grad()
            x, y = _prepare_batch(batch, device=device)
            y_pred, rep = self.model(x, output='both')
            loss_target = F.cross_entropy(y_pred, y)
            loss = loss_target

            if self.cfg['contrastive_loss'] == 'moon' and self.prev_model:
                rep_global = global_model(x, output='rep_only')
                rep_prev = self.prev_model(x, output='rep_only')

                pos = F.cosine_similarity(rep, rep_global)
                neg = F.cosine_similarity(rep, rep_prev)

                logits = torch.cat((pos, neg), dim=1)
                logits /= self.cfg['contrastive_loss_temperature']
                labels = torch.zeros(x.size(0), device=device).long()

                loss_moon = F.cross_entropy(logits, labels)
                loss += self.cfg['contrastive_loss_weight'] * loss_moon

            loss.backward()
            self.optimizer.step()
            return loss
        trainer_colab = Engine(train_collab)

        with trainer_colab.add_event_handler(Events.EPOCH_COMPLETED, self.evaluate,
                                            "private_training", self.private_dls):
            trainer_colab.run(self.private_dl, self.cfg['private_training_epochs'])
        
        res = self.coarse_eval(self.private_dls)

        self.teardown(save_optimizer=True)
        if global_model:
            global_model = global_model.cpu() # TODO is this needed?
        if self.cfg['keep_prev_model']:
            # self.prev_model = self.prev_model.cpu() # TODO is this needed?
            prev_model = copy.deepcopy(self.model)
        return res


def main():
    global cfg
    with open('config_test.py') as f:
        exec(f.read())

    # wandb.tensorboard.patch(root_logdir="wandb/latest-run/files")
    wandb.init(project='master-fed', entity='maschm',
               # group=cfg['group'], job_type="master", name=cfg['group'],
               config=cfg, config_exclude_keys=cfg['ignore'],
               sync_tensorboard=True)
    # wandb.save("./*/*", wandb.run.dir, 'end')
    cfg['path'] = Path(wandb.run.dir)

    if cfg['dataset'] == 'CIFAR100' or cfg['dataset'] == 'CIFAR':
        import CIFAR as Data
    else:
        raise NotImplementedError(f"dataset '{cfg['dataset']}' is unknown")

    private_partial_idxs = load_idx_from_artifact(
        np.array(Data.private_train_data.targets),
        cfg['parties'],
        cfg['subclasses'],
        cfg['samples_per_class'],
        cfg['concentration']
    )
    private_partial_dls, private_combined_dl, private_test_dl = build_private_dls(
        Data.private_train_data,
        Data.private_test_data,
        10,
        cfg['subclasses'],
        private_partial_idxs,
        cfg['init_private_batch_size']
    )
    public_train_dl = DataLoader(Data.public_train_data,
                                 batch_size=cfg['init_public_batch_size'])
    public_test_dl = DataLoader(Data.public_test_data,
                                batch_size=cfg['init_public_batch_size'])

    print(f"train {cfg['parties']} models on")
    subclass_names = [Data.private_train_data.classes[x] for x in cfg['subclasses']]
    combined_class_names = Data.public_train_data.classes + subclass_names
    print("subclasses: ", subclass_names)
    print("all classes: ", combined_class_names)

    with Pool(cfg['pool_size'], init_pool_process,
              [dill.dumps(private_partial_dls),
               dill.dumps(private_combined_dl),
               dill.dumps(private_test_dl),
               dill.dumps(public_train_dl),
               dill.dumps(public_test_dl)]) as pool:
        args = []
        for i in range(cfg['parties']):
            w_cfg = cfg.copy()
            w_cfg['rank'] = i
            w_cfg['path'] = cfg['path'] / str(i)
            w_cfg['model'] = cfg['model_mapping'][i]
            w_cfg['architecture'] = FedMD.FedMD_CIFAR.hyper[w_cfg['model']]
            model = FedMD.FedMD_CIFAR(*w_cfg['architecture'],
                                      projection_size = cfg['projection_head'],
                                      n_classes = 10+len(cfg['subclasses']),
                                      input_size = (3, 32, 32))
            args.append((w_cfg, model))
        workers = pool.starmap(FedWorker, args)
        del args

        if "init_public" in cfg['stages']:
            print("All parties starting with 'init_public'")
            res = pool.map(FedWorker.init_public, workers)
            [workers, res] = list(zip(*res))
            [acc, loss] = list(zip(*res))
            wandb.run.summary["init_public/acc"] = np.average(acc)
            wandb.run.summary["init_public/loss"] = np.average(loss)

            paths = [w.cfg['path'] for w in workers]
            util.save_models_to_artifact(cfg, workers, (acc,loss), "init_public")
        elif "load_init_public" in cfg['stages']:
            util.load_models_from_artifact(cfg, workers, "init_public")

        if "init_private" in cfg['stages']:
            print("All parties starting with 'init_private'")
            res = pool.map(FedWorker.init_private, workers)
            [workers, res] = list(zip(*res))
            [acc, loss] = list(zip(*res))
            wandb.run.summary["init_private/acc"] = np.average(acc)
            wandb.run.summary["init_private/loss"] = np.average(loss)

            paths = [w.cfg['path'] for w in workers]
            util.save_models_to_artifact(cfg, workers, (acc,loss), "init_private")
        elif "load_init_private" in cfg['stages']:
            util.load_models_from_artifact(cfg, workers, "init_private")


        if "collab" in cfg['stages']:
            print("All parties starting with 'collab'")
            res = pool.map(FedWorker.start_collab, workers)
            [workers, res] = list(zip(*res))
            [acc, loss] = list(zip(*res))
            wandb.log({"acc": np.average(acc), "loss": np.average(loss)})
        else:
            return

        global_model = None
        for n in range(cfg['collab_rounds']):
            print(f"All parties starting with collab round [{n+1}/{cfg['collab_rounds']}]")
            alignment_data, avg_logits = None, None
            if cfg['alignment_mode'] != "none":
                if cfg['alignment_mode'] == "public":
                    print(f"Alignment Data: {cfg['num_alignment']} random examples from the public dataset")
                    alignment_idx = np.random.choice(len(Data.public_train_data),
                                                     cfg['num_alignment'],
                                                     replace = False)
                    alignment_dl = DataLoader(Subset(Data.public_train_data,
                                                     alignment_idx),
                                              batch_size=cfg['num_alignment'])
                    alignment_data, alignment_labels = next(iter(alignment_dl))
                elif cfg['alignment_mode'] == "random":
                    print(f"Alignment Data: {cfg['num_alignment']} random noise inputs")
                    alignment_data = torch.rand([cfg['num_alignment']]
                                                + list(Data.private_train_data[0][0].shape))
                else:
                    raise NotImplementedError(f"alignment_mode '{cfg['alignment_mode']}' is unknown")

                logits = pool.starmap(FedWorker.get_logits,
                                   zip(workers, repeat(alignment_data)))
                # [workers, logits] = list(zip(*res))

                avg_logits = torch.zeros_like(logits[0])
                for l in logits:
                    avg_logits += l
                avg_logits /= len(logits)

            res = pool.starmap(FedWorker.collab_round,
                               zip(workers,
                                   repeat(alignment_data),
                                   repeat(avg_logits),
                                   repeat(global_model if cfg['send_global'] else None)))
            [workers, res] = list(zip(*res))

            if cfg['model_averaging']:
                model_global = avg_params([w.model for w in workers])
                print("model parameters averaged")

                evaluator = create_supervised_evaluator(
                    model_global.to(device),
                    {"acc": metrics.Accuracy(),
                     "loss": metrics.Loss(nn.CrossEntropyLoss())},
                    device)
                evaluator.run(private_test_dl)
                wandb.log(evaluator.state.metrics)
            else:
                [acc, loss] = list(zip(*res))
                wandb.log({"acc": np.average(acc), "loss": np.average(loss)})

        if "upper" in cfg['stages']:
            print("All parties starting with 'upper'")
            res = pool.map(FedWorker.upper_bound, workers)
            [workers, res] = list(zip(*res))
            [acc, loss] = list(zip(*res))
            wandb.run.summary["upper/acc"] = np.average(acc)
            wandb.run.summary["upper/loss"] = np.average(loss)

        if "lower" in cfg['stages']:
            print("All parties starting with 'lower'")
            res = pool.map(FedWorker.lower_bound, workers)
            [workers, res] = list(zip(*res))
            [acc, loss] = list(zip(*res))
            wandb.run.summary["lower/acc"] = np.average(acc)
            wandb.run.summary["lower/loss"] = np.average(loss)


    globs = (w.cfg['path'].glob("events.out.tfevents*") for w in workers)
    util.reduce_tb_events_from_globs(globs,str(cfg['path'] / "all"),
                                     reduce_ops = ["mean", "min", "max"])

    wandb.finish()


if __name__ == '__main__':
    mp.set_start_method('spawn')  #, force=True)
    main()
