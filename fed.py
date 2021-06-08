import functools
from itertools import repeat
import logging
import copy
import os
from pathlib import Path
import shutil

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
                           create_supervised_evaluator)
import thop

import FedMD
from data import load_idx_from_artifact, build_private_dls
import util
from nn import (KLDivSoftmaxLoss, avg_params, optim_to, prepare_batch)


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
        os.makedirs(self.cfg['tmp'])
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

        torch.save(self.model.state_dict(), f"{self.cfg['tmp']}/init_public.pth")
        torch.save(self.optimizer.state_dict(), f"{self.cfg['tmp']}/init_public_optim.pth")

        res = 0, 0
        if self.cfg['init_public_epochs'] > 0:
            res = self.evaluator.state.metrics['acc'], self.evaluator.state.metrics['loss']
        self.teardown()
        return res


    @self_dec
    def init_private(self):
        print(f"party {self.cfg['rank']}: start 'init_private' stage")
        self.model.load_state_dict(torch.load(f"{self.cfg['tmp']}/init_public.pth"))
        self.gstep = 0
        self.setup(torch.load(f"{self.cfg['tmp']}/init_public_optim.pth"))

        with self.trainer.add_event_handler(Events.EPOCH_COMPLETED, self.evaluate,
                                            "init_private", self.private_dls):
            self.trainer.run(self.private_dl, self.cfg['init_private_epochs'])

        torch.save(self.model.state_dict(), f"{self.cfg['tmp']}/init_private.pth")
        torch.save(self.optimizer.state_dict(), f"{self.cfg['tmp']}/init_private_optim.pth")

        res = 0, 0
        if self.cfg['init_public_epochs'] > 0:
            res = self.evaluator.state.metrics['acc'], self.evaluator.state.metrics['loss']
        self.teardown()
        return res


    @self_dec
    def upper_bound(self):
        self.model.load_state_dict(torch.load(f"{self.cfg['tmp']}/init_public.pth"))
        self.gstep = 0
        self.setup(torch.load(f"{self.cfg['tmp']}/init_public_optim.pth"))

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
        self.model.load_state_dict(torch.load(f"{self.cfg['tmp']}/init_private.pth"))
        self.gstep = 0
        self.setup(torch.load(f"{self.cfg['tmp']}/init_private_optim.pth"))

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
        self.model.load_state_dict(torch.load(f"{self.cfg['tmp']}/init_private.pth"))
        self.setup(torch.load(f"{self.cfg['tmp']}/init_private_optim.pth"))
        print(f"party {self.cfg['rank']}: start 'collab' stage")

        gstep = self.cfg['init_private_epochs']
        res = self.coarse_eval(self.private_dls)

        self.teardown(save_optimizer=True)
        return res


    def get_alignment(self, alignment_data):
        self.setup(writer=False)
        def collect_alignment(engine, batch):
            self.model.train()
            x = batch[0].to(device)
            with torch.no_grad():
                logits, rep = self.model(x, output="both")
                if self.cfg['alignment_target'] == 'rep' \
                   or self.cfg['alignment_target'] == 'both':
                    if self.cfg['alignment_loss'] == 'KL':
                        rep = F.softmax(rep)
                    engine.state.rep = torch.cat((engine.state.rep, rep.cpu()), dim=0)
                if self.cfg['alignment_target'] == 'logits' \
                   or self.cfg['alignment_target'] == 'both':
                    logits = logits / self.cfg['alignment_temperature']
                    if self.cfg['alignment_loss'] == 'KL':
                        logits = F.softmax(logits)
                    engine.state.logits = torch.cat((engine.state.logits, logits.cpu()), dim=0)

        alignment_ds = DataLoader(TensorDataset(alignment_data),
                                  batch_size=self.cfg['alignment_matching_batchsize'])
        alignment_collector = Engine(collect_alignment)
        alignment_collector.state.logits = torch.tensor([])
        alignment_collector.state.rep = torch.tensor([])
        alignment_collector.run(alignment_ds)

        self.teardown()

        if self.cfg['alignment_target'] == 'logits':
            return alignment_collector.state.logits, 
        elif self.cfg['alignment_target'] == 'rep':
            return alignment_collector.state.rep, 
        elif self.cfg['alignment_target'] == 'both':
            return alignment_collector.state.logits, alignment_collector.state.rep
        else:
            raise NotImplementedError(f"alignment_target '{self.cfg['alignment_target']}' is unknown")


    def alignment(self, alignment_data, alignment_target):
        if self.cfg['alignment_loss'] == "MSE":
            alignment_loss_fn = nn.MSELoss()
        if self.cfg['alignment_loss'] == "L1":
            alignment_loss_fn = nn.L1Loss()
        if self.cfg['alignment_loss'] == "SmoothL1":
            alignment_loss_fn = nn.SmoothL1Loss()
        if self.cfg['alignment_loss'] == "KL":
            alignment_loss_fn = KLDivSoftmaxLoss()

        def train_alignment(engine, batch):
            self.model.train()
            self.optimizer.zero_grad()
            x, target, *rest = prepare_batch(batch, device=device)
            local_logits, local_rep = self.model(x, output="both")
            loss = 0
            if self.cfg['alignment_target'] == 'logits' \
                or self.cfg['alignment_target'] == 'both':
                pred = local_logits / self.cfg['alignment_temperature']
                loss += alignment_loss_fn(pred, target)
            if self.cfg['alignment_target'] == 'both':
                [target] = rest
            if self.cfg['alignment_target'] == 'rep' \
                or self.cfg['alignment_target'] == 'both':
                pred = local_rep / self.cfg['alignment_temperature']
                loss += alignment_loss_fn(pred, target)
            loss.backward()
            self.optimizer.step()
            return loss

        def train_alignment_contrastive(engine, batch):
            self.model.train()
            self.optimizer.zero_grad()
            x, logits, reps = [x.to(device) for x in batch]
            logits_pred, local_rep = self.model(x, output="both")
            logits_pred /= self.cfg['alignment_temperature']
            loss = alignment_loss_fn(logits_pred, logits)

            print(local_rep.shape, reps.shape)
            if self.cfg['alignment_loss'] == 'distillation':
                # TODO contrastive distillation
                pass

            loss.backward()
            self.optimizer.step()
            return loss

        if self.cfg['alignment_loss'] == "contrastive":
            alignment_tr = Engine(train_alignment_rep)
        else:
            alignment_tr = Engine(train_alignment)

        if self.cfg['alignment_target'] == 'both':
            alignment_ds = TensorDataset(alignment_data,
                                         alignment_target[0], alignment_target[1])
        else:
            alignment_ds = TensorDataset(alignment_data, alignment_target)
        # return train_alignment, 
        alignment_dl = DataLoader(alignment_ds,
                                  batch_size=self.cfg['alignment_matching_batchsize'])
        with alignment_tr.add_event_handler(Events.EPOCH_COMPLETED,
                                                    self.evaluate,
                                                    "alignment", self.private_dls):
            alignment_tr.run(alignment_dl, self.cfg['alignment_matching_epochs'])


    @self_dec
    def collab_round(self, alignment_data = None, alignment_target = None,
                     global_model = None):
        self.setup(self.optim_state)

        if alignment_data != None and alignment_target != None:
            self.alignment(alignment_data, alignment_target)


        if global_model:
            global_model = global_model.to(device)
            global_model.eval()
        if self.prev_model:
            self.prev_model = self.prev_model.to(device)
            self.prev_model.eval()

        def train_collab(engine, batch):
            self.model.train()
            self.optimizer.zero_grad()
            x, y = prepare_batch(batch, device=device)
            y_pred, rep = self.model(x, output='both')
            loss_target = F.cross_entropy(y_pred, y)
            loss = loss_target

            if self.cfg['contrastive_loss'] == 'moon' and self.prev_model:
                rep_global = global_model(x, output='rep_only')
                rep_prev = self.prev_model(x, output='rep_only')

                pos = F.cosine_similarity(rep, rep_global).reshape(-1,1)
                neg = F.cosine_similarity(rep, rep_prev).reshape(-1,1)

                logits = torch.cat((pos, neg), dim=1)
                logits /= self.cfg['contrastive_loss_temperature']

                # first "class" sim(rep_global) is the ground truth
                labels = torch.zeros(x.size(0), device=device).long()
                loss_moon = F.cross_entropy(logits, labels)
                loss += self.cfg['contrastive_loss_weight'] * loss_moon

            loss.backward()
            self.optimizer.step()
            return loss
        collab_tr = Engine(train_collab)

        with collab_tr.add_event_handler(Events.EPOCH_COMPLETED, self.evaluate,
                                            "private_training", self.private_dls):
            collab_tr.run(self.private_dl, self.cfg['private_training_epochs'])
        
        res = self.coarse_eval(self.private_dls)

        self.teardown(save_optimizer=True)
        if self.cfg['keep_prev_model']:
            del self.prev_model
            self.prev_model = copy.deepcopy(self.model)
        return res



def main():
    global cfg
    with open('config_test.py') as f:
        exec(f.read())

    if cfg['variant'] == 'fedmd':
        cfg['global_model'] = 'none'
        cfg['replace_local_model'] = False
        cfg['keep_prev_model'] = False
        cfg['send_global'] = False
        cfg['contrastive_loss'] = 'none'
    elif cfg['variant'] == 'fedavg':
        cfg['global_model'] = 'averaging'
        cfg['replace_local_model'] = True
        cfg['keep_prev_model'] = False
        cfg['send_global'] = False
        cfg['alignment_data'] = 'none'
        cfg['contrastive_loss'] = 'none'
    elif cfg['variant'] == 'moon':
        cfg['global_model'] = 'averaging'
        cfg['replace_local_model'] = True
        cfg['keep_prev_model'] = True
        cfg['send_global'] = True
        cfg['alignment_data'] = 'none'
        cfg['contrastive_loss'] = 'moon'

    # wandb.tensorboard.patch(root_logdir="wandb/latest-run/files")
    wandb.init(project='master-fed', entity='maschm',
               # group=cfg['group'], job_type="master", name=cfg['group'],
               config=cfg, config_exclude_keys=cfg['ignore'],
               sync_tensorboard=True)
    # wandb.save("./*/*", wandb.run.dir, 'end')
    cfg['path'] = Path(wandb.run.dir)
    cfg['tmp'] = Path("./wandb/tmp/")
    shutil.rmtree(cfg['tmp'], ignore_errors=True)

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
            w_cfg['tmp'] = cfg['tmp'] / str(i)
            w_cfg['model'] = cfg['model_mapping'][i]
            w_cfg['architecture'] = FedMD.FedMD_CIFAR.hyper[w_cfg['model']]
            model = FedMD.FedMD_CIFAR(*w_cfg['architecture'],
                                      projection = cfg['projection_head'],
                                      n_classes = 10+len(cfg['subclasses']),
                                      input_size = (3, 32, 32))
            args.append((w_cfg, model))
        workers = pool.starmap(FedWorker, args)
        del args

        model = copy.deepcopy(workers[0].model)
        input = Data.public_train_data[0][0].unsqueeze(0)
        macs, params = thop.profile(model, inputs=(input, ))
        print(*thop.clever_format([macs, params], "%.3f"))
        wandb.config.update({"model": {"macs": macs, "params": params}})
        del model

        if "init_public" in cfg['stages']:
            print("All parties starting with 'init_public'")
            res = pool.map(FedWorker.init_public, workers)
            [workers, res] = list(zip(*res))
            [acc, loss] = list(zip(*res))
            acc, loss = np.average(acc), np.average(loss)
            wandb.run.summary["init_public/acc"] = acc
            wandb.run.summary["init_public/loss"] = loss
            if "save_init_public" in cfg['stages']:
                util.save_models_to_artifact(cfg, workers, "init_public",
                                             {"acc": acc, "loss": loss})
        elif "load_init_public" in cfg['stages']:
            util.load_models_from_artifact(cfg, workers, "init_public")

        if "init_private" in cfg['stages']:
            print("All parties starting with 'init_private'")
            res = pool.map(FedWorker.init_private, workers)
            [workers, res] = list(zip(*res))
            [acc, loss] = list(zip(*res))
            acc, loss = np.average(acc), np.average(loss)
            wandb.run.summary["init_private/acc"] = acc
            wandb.run.summary["init_private/loss"] = loss
            if "save_init_private" in cfg['stages']:
                util.save_models_to_artifact(cfg, workers, "init_private",
                                             {"acc": acc, "loss": loss})
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
        if cfg['global_model'] != 'none':
            global_model = copy.deepcopy(workers[0].model)

        for n in range(cfg['collab_rounds']):
            print(f"All parties starting with collab round [{n+1}/{cfg['collab_rounds']}]")
            alignment_data, avg_logits = None, None
            if cfg['alignment_data'] != "none":
                if cfg['alignment_data'] == "public":
                    print(f"Alignment Data: {cfg['num_alignment']} random examples from the public dataset")
                    alignment_idx = np.random.choice(len(Data.public_train_data),
                                                     cfg['num_alignment'],
                                                     replace = False)
                    alignment_dl = DataLoader(Subset(Data.public_train_data,
                                                     alignment_idx),
                                              batch_size=cfg['num_alignment'])
                    alignment_data, alignment_labels = next(iter(alignment_dl))
                elif cfg['alignment_data'] == "random":
                    print(f"Alignment Data: {cfg['num_alignment']} random noise inputs")
                    alignment_data = torch.rand([cfg['num_alignment']]
                                                + list(Data.private_train_data[0][0].shape))
                else:
                    raise NotImplementedError(f"alignment_data '{cfg['alignment_data']}' is unknown")

                res = pool.starmap(FedWorker.get_alignment,
                                   zip(workers, repeat(alignment_data)))
                alignment_targets, *rest = list(zip(*res))

                avg_alignment_targets = torch.zeros_like(alignment_targets[0])
                for t in avg_alignment_targets:
                    avg_alignment_targets += t.clone()
                avg_alignment_targets /= len(avg_alignment_targets)

                if cfg['alignment_target'] == "both":
                    [reps] = rest
                    avg_reps = torch.zeros_like(reps[0])
                    for r in reps:
                        avg_reps += r
                    avg_reps /= len(reps)
                    avg_alignment_targets = avg_alignment_targets, avg_reps

            res = pool.starmap(FedWorker.collab_round,
                               zip(workers,
                                   repeat(alignment_data),
                                   repeat(avg_alignment_targets),
                                   repeat(global_model if cfg['send_global'] else None)))
            [workers, res] = list(zip(*res))

            if cfg['global_model'] == 'averaging':
                global_weights = avg_params([w.model for w in workers])
                global_model.load_state_dict(global_weights)
                print("model parameters averaged")
            elif cfg['global_model'] == 'distillation':
                # TODO
                pass

            if cfg['global_model'] != 'none':
                evaluator = create_supervised_evaluator(
                    global_model.to(device),
                    {"acc": metrics.Accuracy(),
                     "loss": metrics.Loss(nn.CrossEntropyLoss())},
                    device)
                evaluator.run(private_test_dl)
                acc = evaluator.state.metrics['acc']
                loss = evaluator.state.metrics['loss']
                wandb.log({"acc": np.average(acc), "loss": np.average(loss)})
                global_model = global_model.cpu()
            else:
                [acc, loss] = list(zip(*res))
                acc, loss = np.average(acc), np.average(loss)
                wandb.log({"acc": acc, "loss": loss})

            if cfg['replace_local_model']:
                if global_model is None:
                    raise Execption("Global model is None. Can't replace local models")
                for w in workers:
                    w.model.load_state_dict(global_model.state_dict())
                print("local models replaced")

        if "save_final" in cfg['stages']:
            for w in workers:
                torch.save(w.model.state_dict(), f"{w.cfg['tmp']}/final.pth")
                torch.save(None, f"{w.cfg['tmp']}/final_optim.pth")
            util.save_models_to_artifact(cfg, workers, "final",
                                         {"acc": acc, "loss": loss})

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
    # logger = mp.log_to_stderr()
    # logger.setLevel(multiprocessing.SUBDEBUG)
    mp.set_start_method('spawn')  #, force=True)
    main()
