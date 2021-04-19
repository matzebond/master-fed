import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import ignite
from ignite import metrics
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
import wandb
from itertools import *
from multiprocessing import Barrier, Process, Pipe, current_process
from multiprocessing.connection import wait
import dill
import os
from pathlib import Path
from time import sleep
import logging

import CIFAR
import FedMD
from data import load_idx_from_artifact, build_private_dls

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

def run(m_id, r_pipe, w_pipe, barrier):
    cfg = r_pipe.recv()
    w_pipe.send(f"hello from {os.getpid()}")

    private_dl = dill.loads(r_pipe.recv())
    private_combined_dl = dill.loads(r_pipe.recv())
    private_test_dl = dill.loads(r_pipe.recv())
    public_train_dl = dill.loads(r_pipe.recv())
    public_test_dl = dill.loads(r_pipe.recv())

    np.random.seed()
    torch.manual_seed(np.random.randint(0, 0xffff_ffff))

    print(f"start run {cfg['rank']} in pid {os.getpid()}")

    cfg['architecture'] = FedMD.FedMD_CIFAR_hyper[cfg['model']]
    model = FedMD.FedMD_CIFAR(10+len(cfg['subclasses']),
                              (3,32,32),
                              *cfg['architecture']).to(device)

    run = wandb.init(project='mp-test', entity='maschm',
                     group=cfg['group'], job_type='party',
                     config=cfg, config_exclude_keys=['main_id', 'group', 'rank'],
                     sync_tensorboard=True)
    writer = SummaryWriter(wandb.run.dir)
    wandb.watch(model)

    global gstep
    gstep = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['public_lr'])

    trainer = create_supervised_trainer(
        model,
        optimizer,
        nn.CrossEntropyLoss(),
        device,
    )

    trainer_logit = create_supervised_trainer(
        model,
        optimizer,
        nn.MSELoss(), ## TODO use KL Loss
        device,
    )


    loss_metric = metrics.Loss(nn.CrossEntropyLoss())
    evaluator = create_supervised_evaluator(model, metrics={"acc": metrics.Accuracy(),
                                                            "loss": loss_metric})
    def log_metrics(engine, trainer, title):
        epoch = trainer.state.epoch
        max_epochs = trainer.state.max_epochs
        acc = engine.state.metrics['acc']
        loss = engine.state.metrics['loss']
        print(f"{title} - acc: {acc:.3f} loss: {loss:.4f}")
        writer.add_scalar(f"{title}/acc", acc, gstep)
        writer.add_scalar(f"{title}/loss", loss, gstep)


    def evaluate(trainer, stage="unknown stage", dls={}, advance=True):
        epoch = trainer.state.epoch
        max_epochs = trainer.state.max_epochs
        print(f"{stage} [{epoch:2d}/{max_epochs:2d}]")
        for name, dl in dls.items():
            with evaluator.add_event_handler(Events.COMPLETED,
                                             log_metrics, trainer, name):
                evaluator.state.max_epochs = None
                evaluator.run(dl)
        if advance:
            global gstep
            gstep += 1

    public_dls =  {"public_train", public_train_dl,
                   "public_test", public_test_dl}
    private_dls = {"private_train": private_dl,
                   "private_test": private_test_dl}



    # @trainer.on(Events.ITERATION_COMPLETED)
    def log_training(engine):
        batch_loss = engine.state.output
        e = engine.state.epoch
        n = engine.state.max_epochs
        i = engine.state.iteration
        print(f"Epoch {e}/{n}: {i} - batch loss: {batch_loss:.4f}")


    barrier.wait()
    print(f"party {cfg['rank']}: start initial public training")
    with trainer.add_event_handler(Events.EPOCH_COMPLETED, evaluate,
                                   "public_init", public_dls):
        trainer.run(public_train_dl, cfg['public_epochs'])


    barrier.wait()
    print(f"party {cfg['rank']}: start initial private training")
    with trainer.add_event_handler(Events.EPOCH_COMPLETED, evaluate,
                                   "private_init", private_dls):
        trainer.run(private_dl, cfg['private_init_epochs'])

    barrier.wait()
    for collab_idx in range(cfg['collab_rounds']):
        if cfg['alignment_mode'] != None:
            alignment_data = r_pipe.recv()
            with torch.no_grad():
                logits = model(alignment_data)
                w_pipe.send(logits)

            barrier.wait()
            logits = r_pipe.recv()
            logit_dl = DataLoader(TensorDataset(alignment_data, logits),
                                batch_size=cfg['logits_matching_batchsize'])
            with trainer_logit.add_event_handler(Events.EPOCH_COMPLETED, evaluate,
                                                 "alignment", private_dls):
                trainer_logit.run(logit_dl, cfg['logits_matching_epochs'])


        with trainer.add_event_handler(Events.EPOCH_COMPLETED, evaluate,
                                       "private_training", private_dls):
            trainer.run(private_dl, cfg['private_training_epochs'])


        if cfg['model_averaging']:
            pass


        with evaluator.add_event_handler(Events.COMPLETED,
                                         log_metrics, trainer, "coarse/private_test"):
            evaluator.state.max_epochs = None
            evaluator.run(private_test_dl)


    r_pipe.close()
    w_pipe.close()
    run.finish()


    

def main():
    cfg = {
        'main_id': wandb.util.generate_id(),
        'group': wandb.util.generate_id(),
        'samples_per_class': 3,
        'dataset': 'CIFAR100',
        'data_variant': 'iid',
        'subclasses': [0,2,20,63,71,82],
        'parties': 3,
        'load_private_idx': True,
        'optim': 'Adam',
        'public_lr': 0.001,
        'public_epochs': 0,
        'public_batch_size': 64,
        'private_init_epochs': 10,
        'private_init_batch_size': 32,
        'collab_rounds': 10,
        'alignment_mode': 'public',
        'num_alignment': 100,
        'logits_matching_epochs': 10,
        'logits_matching_batchsize': 128,
        'private_training_epochs': 1,
        # 'private_training_batchsize': 5, # TODO not supported
        'model_averaging': False,
        'stages': ['pre-public', 'pre-private', 'init-private', 'colab'],
    }

    model_mapping =  list(islice(cycle(range(len(FedMD.FedMD_CIFAR_hyper))), cfg['parties']))
    # model_mapping = list(repeat(4, cfg['parties']))


    processes = []
    pipes = []
    barrier = Barrier(cfg['parties']+1)

    for i in range(cfg['parties']):
        p_cfg = cfg.copy()
        p_cfg['rank'] = i
        p_cfg['model'] = model_mapping[i]

        p_recv, m_send = Pipe(duplex=False)
        m_recv, p_send = Pipe(duplex=False)
        p = Process(target=run, args=(i, p_recv, p_send, barrier))
        p.start()

        m_send.send(p_cfg)
        print(m_recv.recv())

        p_recv.close()
        p_send.close()
        processes.append(p)
        pipes.append((m_recv, m_send))

    wandb.init(project='mp-test', entity='maschm',
               group=cfg['group'], job_type="master", id=cfg['main_id'],
               config=cfg, config_exclude_keys=['main_id', 'group', 'rank', 'model'])

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
        cfg['private_init_batch_size']
    )
    public_train_dl = DataLoader(CIFAR.public_train_data,
                                 batch_size=cfg['public_batch_size'])
    public_test_dl = DataLoader(CIFAR.public_test_data,
                                batch_size=cfg['public_batch_size'])

    print(f"train {cfg['parties']} models on")
    subclass_names = list(map(lambda x: CIFAR.private_train_data.classes[x],
                              cfg['subclasses']))
    combined_class_names = CIFAR.public_train_data.classes + subclass_names
    print("subclasses: ", subclass_names)
    print("all classes: ", combined_class_names)


    for _, send in pipes:
        send.send(dill.dumps(private_partial_dls[i]))
        send.send(dill.dumps(private_combined_dl))
        send.send(dill.dumps(private_test_dl))
        send.send(dill.dumps(public_train_dl))
        send.send(dill.dumps(public_test_dl))


    barrier.wait()
    print("All parties start with initial public training")
    barrier.wait()
    print("All parties start with initial private training")
    barrier.wait()



    for n in range(cfg['collab_rounds']):
        if cfg['alignment_mode'] != "none":
            # select alingment data
            if cfg['alignment_mode'] == "public":
                print(f"Alignment Data: {cfg['num_alignment']} random examples from the public dataset")
                alignment_idx = np.random.choice(len(CIFAR.public_train_data),
                                                cfg['num_alignment'], replace = False)
                alignment_dataset = DataLoader(Subset(CIFAR.public_train_data, alignment_idx),
                                            batch_size=cfg['num_alignment'])
                alignment_data, alignment_labels = next(iter(alignment_dataset))
            elif cfg['alignment_mode'] == "random":
                print(f"Alignment Data: {cfg['num_alignment']} random noise inputs")
                alignment_data = torch.rand([cfg['num_alignment']]
                                            + list(private_train_data[0][0].shape))
            else:
                raise NotImplementedError(f"alignment_mode '{cfg['alignment_mode']}' is unknown")

            for _, send in pipes:
                send.send(alignment_data)

            print(f"main: alignments for round {n} send")

            logits = 0
            for i,(recv,_) in enumerate(pipes):
                logits += recv.recv()
            logits /= cfg['parties']

            print(f"main: logits for round {n} received")
            barrier.wait()

            for _, send in pipes:
                send.send(logits)

            print(f"main: done with alignment for round {n}")


        if cfg['model_averaging']:
            pass



    for p in processes:
        p.join()
    wandb.finish()


if __name__ == '__main__':
    main()
