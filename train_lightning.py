#!python
from __future__ import division
import faulthandler; faulthandler.enable()

import inspect
import json
import os
import shutil
import signal
import sys
import uuid
from copy import deepcopy

import pytorch_lightning as pl
from matplotlib import pyplot as plt
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.metrics import Metric, R2Score, ExplainedVariance
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from sacred import Experiment
from dataset_ingredient import data_ingredient, load_data
import subprocess

from factor_gnn.models import *
from factor_gnn.batch import Batch
from factor_gnn.datasets import *


class Averager(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("size", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, value: torch.Tensor, B: int):
        self.sum += value.sum()
        self.size += value.numel()*B

    def compute(self):
        return self.sum.float()/self.size


def add_prefix(d, prefix=""):
    if len(prefix.strip())==0:
        return d
    newd = {}
    for k in d.keys():
        newk = prefix + "_" + k
        newd[newk] = d[k]
    return newd


class RegressionScatterPlotCallback(Callback):
    def __init__(
            self,
            plot_interval: int = 0,  # 0 means don't plot
    ):
        self.plot_interval = plot_interval

    def on_validation_epoch_end(self, trainer, system):
        if self.plot_interval==0:
            pass
        if (trainer.current_epoch + 1)%self.plot_interval==0:
            self._scatter_plot

    def _scatter_plot(self):
        pass

    def on_test_epoch_end(self, trainer, system):
        pass


class FactorGraphDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = self.args["batch_size"]
        self.setup()

    def prepare_data(self):
        # called only on 1 GPU
        # already prepared somewhere else
        pass

    def setup(self, stage: Optional[str] = None):
        self.train, self.val, self.test = load_data(**self.args)[0][:3]

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, collate_fn=lambda x:Batch.from_data_list(x), )

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=lambda x:Batch.from_data_list(x), )

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=lambda x:Batch.from_data_list(x), )


class GNNSystem(pl.LightningModule):
    def __init__(
            self, model_kwargs, control_kwargs, architecture="rnn"):
        super().__init__()
        self.architecture = architecture
        if architecture=="rnn":
            self.model = MultiGATConvGRU(**model_kwargs)
        if architecture=="stack":
            self.model = MultiGATConvStack(**model_kwargs)

        # Metrics
        self.train_loss = Averager()
        self.val_loss = Averager()
        self.test_loss = Averager()
        if architecture in ['rnn', 'ode', 'stack']:
            metrics = {
                "r2":ExplainedVariance(),
            }
            self.train_metrics = pl.metrics.MetricCollection(add_prefix(deepcopy(metrics), "train"))
            self.val_metrics = pl.metrics.MetricCollection(add_prefix(deepcopy(metrics), "val"))
            self.test_metrics = pl.metrics.MetricCollection(add_prefix(deepcopy(metrics), "test"))

        for k in control_kwargs.keys():
            setattr(self, k, control_kwargs[k])

        if self.loss_fn=='mse':
            self.criterion = nn.MSELoss(reduction='mean')
        elif self.loss_fn=='mae':
            self.criterion = nn.L1Loss(reduction='mean')
        elif self.loss_fn=='bce':
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            raise NotImplementedError(f"loss_fn {self.loss_fn} not supported")

        if self.plot_interval==0:
            self.plot_interval = np.iinfo(np.int).max

        self.plot_graph_val = None

        self.y_plot = []
        self.ypred_plot = []

    def forward(self, x, output_dynamics=False, nstep=None):
        if nstep is None:
            nstep = self.get_nstep()
        return self.model(x, output_dynamics=output_dynamics, nstep=nstep)

    def get_nstep_constant(self):
        return self.nstep_start

    def get_nstep_random(self):
        return np.random.randint(self.nstep_start, self.nstep_end + 1)

    def get_nstep_linear(self):
        return int(
            max(self.nstep_start, min(self.nstep_end,
                                      self.nstep_start + (
                                              self.current_epoch - self.burnin_epochs)/self.nstep_linear_epochs*(
                                              self.nstep_end - self.nstep_start)
                                      ))
        )

    def get_nstep(self):
        if self.current_epoch < self.burnin_epochs:
            return self.nstep_start
        if self.nstep_schedule_method=='constant':
            return self.get_nstep_constant()
        elif self.nstep_schedule_method=='random':
            return self.get_nstep_random()
        elif self.nstep_schedule_method=='linear':
            return self.get_nstep_linear()
        else:
            raise NotImplementedError(f"nstep scheduling method {self.nstep_schedule_method} not implemented")

    def _maybe_crop_target(self,yhat,y):
        trainset = self.train_dataloader().dataset
        # trainset = self.trainer._data_connector._train_dataloader_source.dataloader().dataset
        if isinstance(trainset, LDPCDatasetBasic):
            M = trainset.M
            N = trainset.N
            yhat = yhat.view(-1,N,yhat.shape[-1])
            yhat = yhat[:,:M,:].reshape(-1,1).contiguous()
            y = y.view(-1,N,y.shape[-1])
            y = y[:,:M,:].reshape(-1,1).contiguous()
            return yhat, y
        else:
            return yhat,y
    
    def training_step(self, g, batch_idx):
        B = g.y.shape[0]
        g = g.to(next(self.model.parameters()).device)
        yhat = self.model.forward(g, output_dynamics=False, nstep=self.get_nstep())
        yhat, y = self._maybe_crop_target(yhat, g.y)
        B = y.shape[0]
        if self.architecture in ["cnf", "cnf2"]:
            loss = yhat[0].sum()/g.num_graphs
        else:
            loss = self.criterion(yhat, y)
            if self.loss_fn=='bce':  # add -entropy(data) to make loss KL divergence
                if isinstance(self.train_dataloader().dataset, ThirdOrderDatasetBipartite):
                    loss -= torch.nn.functional.binary_cross_entropy(y.sigmoid(), y.sigmoid())
                # loss -= self.criterion(g.y.exp(), g.y.exp())

        # update metrics
        if self.loss_fn == 'bce' and isinstance(self.train_dataloader().dataset, LDPCDatasetBasic):
            prediction = (yhat > 0).long()   # (M * B, 1)
            error = (prediction - y).abs().mean()
            # self.log("train_loss", self.train_loss(error, 1.0), prog_bar=True,logger=True, on_step=True)
            self.log("train_loss", error, prog_bar=True,logger=True, on_step=True, on_epoch=True)
        else:
            self.log("train_loss", loss * B, prog_bar=True,logger=True, on_step=True, on_epoch=True)
            # self.log("train_loss", self.train_loss(loss, B), prog_bar=True,logger=True, on_step=True)
        if self.architecture in ['rnn', 'ode','stack']:
            if self.loss_fn == 'bce':
                if isinstance(self.train_dataloader().dataset, ThirdOrderDatasetBipartite):
                    self.train_metrics.update(yhat.sigmoid(), y)
                if isinstance(self.train_dataloader().dataset, LDPCDatasetBasic):
                    self.train_metrics.update(prediction, y)
            else:
                self.train_metrics.update(yhat, y)
        return loss

    def train_dataloader(self):
        return self.trainer._data_connector._train_dataloader_source.dataloader()
    def validation_step(self, g, batch_idx):
        if batch_idx==0:
            self.plot_graph_val = g.data_list[0]
        B = g.y.shape[0]
        g = g.to(next(self.model.parameters()).device)
        yhat = self.model.forward(g, output_dynamics=False, nstep=self.get_nstep())
        yhat, y = self._maybe_crop_target(yhat, g.y)
        B = y.shape[0]
        if self.should_plot():
            self.y_plot.append(y.detach().cpu())
            self.ypred_plot.append(yhat.detach().cpu())
        if self.architecture in ["cnf", "cnf2"]:
            loss = yhat[0].sum()/g.num_graphs
        else:
            loss = self.criterion(yhat, y)
            if isinstance(self.train_dataloader().dataset, ThirdOrderDatasetBipartite):
                loss -= torch.nn.functional.binary_cross_entropy(y.sigmoid(), y.sigmoid())
        # update metrics
        if self.loss_fn == 'bce' and isinstance(self.train_dataloader().dataset, LDPCDatasetBasic):
            prediction = (yhat > 0).long()   # (M * B, 1)
            error = (prediction - y).abs().mean()
            # self.log("val_loss", self.val_loss(error, 1.0), prog_bar=True,logger=True, on_step=True, sync_dist=True)
            self.log("val_loss", error, prog_bar=True,logger=True,  sync_dist=True, on_step=False, on_epoch=True)
            loss = error
        else:
            self.log("val_loss", loss*B, prog_bar=True,logger=True, sync_dist=True, on_step=False, on_epoch=True)
        # self.log("val_loss", self.val_loss(loss, B), prog_bar=True, logger=False, sync_dist=True)
        if self.architecture in ['rnn', 'ode','stack']:
            if self.loss_fn == 'bce':
                if isinstance(self.train_dataloader().dataset, ThirdOrderDatasetBipartite):
                    self.val_metrics.update(yhat.sigmoid(), y)
                if isinstance(self.train_dataloader().dataset, LDPCDatasetBasic):
                    self.val_metrics.update(prediction, y)
            else:
                self.val_metrics.update(yhat, y)
        self.log("lr", self.optimizers().optimizer.param_groups[0]['lr'], prog_bar=True, sync_dist=True)
        if self.architecture in ['rnn','stack']:
            self.log("nsetp", self.get_nstep(), prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, g, batch_idx):
        B = g.y.shape[0]
        g = g.to(next(self.model.parameters()).device)
        yhat = self.model.forward(g,output_dynamics=False, nstep=self.get_nstep())
        yhat, y = self._maybe_crop_target(yhat, g.y)
        if self.architecture in ["cnf","cnf2"]:
            loss = yhat[0].sum() / g.num_graphs
        else:
            loss = self.criterion(yhat, y) 
            if self.loss_fn == 'bce':     # add -entropy(data) to make loss KL divergence
                loss -= torch.nn.functional.binary_cross_entropy(y.sigmoid(), y.sigmoid())
        # update metrics
        if self.loss_fn == 'bce' and isinstance(self.train_dataloader().dataset, LDPCDatasetBasic):
            prediction = (yhat > 0).long()   # (M * B, 1)
            error = (prediction - y).abs().mean()
            self.log("test_loss", error, prog_bar=True,logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.val_loss.update(error,1.0)
            loss = error
        else:
            self.log("test_loss", loss, prog_bar=True,logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.val_loss.update(loss, 1.0)
        if self.architecture in ['rnn', 'ode','stack']:
            if self.loss_fn == 'bce':
                self.test_metrics.update(yhat.sigmoid(), y)
            else:
                self.test_metrics.update(yhat, y)
        return loss

    def training_epoch_end(self, outputs):
        # self.log("train_loss_epoch", self.train_loss.compute(), prog_bar=True, logger=True)
        if self.architecture in ['rnn', 'ode','stack']:
            self.log_dict(self.train_metrics.compute(), prog_bar=True)
            self.train_metrics.reset()
        # self.train_loss.reset()

    def validation_epoch_end(self, outputs):
        # self.log("val_loss_epoch", self.val_loss.compute(), prog_bar=True, logger=True, on_step=False)
        self.log("val_loss_epoch", torch.stack(outputs).mean())
        # self.val_loss.reset()
        if self.architecture in ['rnn', 'ode','stack']:
            metrics_value = self.val_metrics.compute()
            self.log_dict(metrics_value, prog_bar=True, logger=True, on_epoch=True)
            if (self.current_epoch +1) %self.plot_interval == 0:
                self._scatter_plot(self.val_metrics['val_r2'], prefix='val')
            self.val_metrics.reset()
        elif self.architecture in ["cnf", 'cnf2']:
            if self.should_plot():
                self._plot_samples(self.plot_graph_val, prefix='val')
                self.y_plot = []
                self.ypred_plot = []
    def should_plot(self):
        return (self.current_epoch + 1)%self.plot_interval==0

    def _scatter_plot(self, metric, prefix='val'):
        """
        Given a certain metric, scatter plot the `y` and `y_pred`
        """
        preds = torch.cat(self.ypred_plot, dim=0).view(-1)
        target = torch.cat(self.y_plot, dim=0).view(-1)
        # take a subsample
        idx = torch.randperm(preds.shape[0])[:self.plot_samples]
        preds_plot = preds[idx]
        target_plot = target[idx]
        plt.scatter(preds_plot.detach().cpu(), target_plot.detach().cpu())
        plt.plot([0.0, 1.0], [0.0, 1.0])
        plt.xlabel("prediction")
        plt.ylabel("target")
        plt.gca().set_aspect('equal')
        r2 = metric.compute()
        plt.legend([f'r2={r2}', ""])
        if isinstance(self.logger, WandbLogger):
            self.log_dict(
                {
                    f"{prefix}_prediction": wandb.Image(plt)
                },
                )
        plt.close()

    def test_epoch_end(self, outputs):
        self.log("test_loss_epoch", self.test_loss.compute(), prog_bar=True, logger=True)
        if self.architecture in ['rnn', 'ode']:
            self._scatter_plot(self.test_metrics['test_r2'], prefix='test')
            self.log_dict(self.test_metrics.compute(), prog_bar=True)
            self.test_metrics.reset()
        self.test_loss.reset()

    def configure_optimizers(self):
        if self.optimizer_name=='adam':
            opt = torch.optim.Adam(self.model.parameters(), lr=self.init_lr)
        elif self.optimizer_name=='sgd':
            opt = torch.optim.SGD(self.model.parameters(), lr=self.init_lr, momentum=0.9, nesterov=True)
        elif self.optimizer_name=='rmsprop':
            opt = torch.optim.RMSprop(self.model.parameters(), lr=self.init_lr, weight_decay=1e-6)
        elif self.optimizer_name=='adadelta':
            opt = torch.optim.Adadelta(self.model.parameters(), lr=self.init_lr, weight_decay=0.0)
        else:
            raise NotImplementedError(f"optimizer name {self.optimizer_name} not supported")

        if hasattr(opt, 'momentum'):
            cycle_momentum = True
        else:
            cycle_momentum = False

        if self.scheduler_name=='onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=self.max_lr,
                steps_per_epoch=len(self.train_dataloader.dataloader),
                epochs=self.epochs,
                cycle_momentum=cycle_momentum,
                pct_start=0.2,
            )
            sch_dic = {
                'scheduler':scheduler,
                'interval':'step',
                'monitor':'val_loss_epoch',
            }

        elif self.scheduler_name=='plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min',
                factor=0.2,
                patience=self.scheduler_patience,
            )
            sch_dic = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'monitor': 'val_loss_epoch',
            }
        elif self.scheduler_name == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                opt, gamma=self.gamma,
                # patience=self.scheduler_patience,
            )
            sch_dic = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'monitor': 'val_loss_epoch',
            }
        else:
            raise NotImplementedError(self.scheduler_name)
        return [opt], [sch_dic]


class SIGTERMERROR(Exception):
    """
    raised when a SIGTERM signal is received
    """
    pass


GIT_HASH = subprocess.check_output(["git", "describe", "--always"]).strip().decode()

ex = Experiment("thirdorder", ingredients=[data_ingredient])


def load_checkpoint(args, state_vars):
    """
    Resume to a previous training state
    """
    model = state_vars["model"]
    model_device = next(model.parameters()).device
    os.chdir(args["realoutdir"])
    with open("checkpoint.pt", "rb") as file:
        data = pickle.load(file)
    model.load_state_dict(data["model_state_dict"].to(model_device))
    state_vars["optimizer"].load_state_dict(data["optimizer_state_dict"])
    state_vars["train_loss"] = data["train_loss"]
    state_vars["val_loss"] = data["val_loss"]
    state_vars["epoch"] = data["epoch"]


def create_checkpoint(state_vars):
    # create a snapshot of training states
    checkpoint = {
        "model_state_dict":{
            k:v.cpu() for (k, v) in state_vars["model"].state_dict().items()
        },
        "optimizer_state_dict":state_vars["optimizer"].state_dict(),
        "epoch":state_vars["epoch"],
        "train_loss":state_vars["train_loss"],
        "val_loss":state_vars["val_loss"],
    }

    return checkpoint


@ex.config
def config_initial():
    # generate a random uuid
    identifier = uuid.uuid4().hex
    args = {"id":identifier}
    model_args = {}


@ex.config
def config_hyperparameter(args, model_args):
    args.update(
        # seed=42,
        epochs=1000,
        burnin_epochs=3,
        checkpoint=20,
        init_lr=1e-3,
        max_lr=1e-1,
        gamma = 0.98,  # decay coefficient for exponeitial learning rate
        weight_decay=1e-6,
        # batch_size = 512,
        patience=10,
        scheduler_patience=5,
        nstep_schedule_method="random",  # ['constant','linear','random']
        bn=True,
        nstep_start=30,  # used in ['linear','random']
        nstep_end=50,  # used in ['linear','random']
        nstep_warmup_epoch=1,  # warm-up epochs using "nstep_start", used in ['linear','random']
        nstep_linear_epochs=50,  # #epochs taken to linearly change from "nstep_start" to "nstep_end"
        optimizer_name='adam',
        scheduler_name='plateau',
        rnn_method='gru',
    )

    model_args.update(
        nhid=64,
        nstate=64,  # hidden state dimension
        damping=0.0,  # smoothing the hidden state update
        heads=3,
        nlayer=3,
        asrnn_kwargs={
            "gamma":0.1,
            "epsilon":1.0
        },
        rnn_method="gru_ln",
        aggregation_method="sum",
        init_method="encode",
        decode_method="varstate_mlp",
        decode_series_len=0,
        const_factor_input=False,
        gat_module='gat',
        add_self_loops=False,
        use_factor_net=True,
        use_bn=False,
        alpha=0.0,
        beta=1.0,
    )


@ex.config
def config_control(args):
    git_hash = GIT_HASH
    args.update(
        architecture="rnn",
        mode="train",  # 'train' or 'eval' or 'lr_finder'
        run_id=0,
        reweight_loss=True,
        loss_fn="mse",
        no_cuda=False,
        fastmode=False,
        record_dynamics=False,  # whether to record recurrent dynamics of latent states
        record_dynamics_interval=10,  # interval of epochs to record latent dynamics
        readout="varmlp",
        nstep_schedule_method="constant", # ['constant','linear','random']
        snap_interval=1,
        snap_state_interval=30,
        plot_interval=5,
        plot_samples=500,
        eval_interval=10,
        igpu=0,
        ipartite=1,  # which part is target
        use_cuda=True,
    )
    args["cuda"] = not args["no_cuda"] and torch.cuda.is_available()
    args["device"] = "cuda:0" if args["cuda"] else ["cpu"]


@ex.config
def config_path(args):
    args.update(
        outdir="outputs/test",
        jobname="default",
    )
    outdir = os.path.join(args["outdir"], args["id"])
    args.update(realoutdir=outdir)
    print(args["realoutdir"])


@ex.config
def config_dataset(args, dataset):
    args['dataset'] = dataset


@ex.capture
def path_init(args, model_args, dataset):
    if os.path.isdir(args["realoutdir"]):
        shutil.rmtree(args["realoutdir"])
    os.makedirs(args["realoutdir"])
    out_dict = {
        'args':args,
        'model_args':model_args,
        'dataset':dataset
    }
    with open(os.path.join(args['realoutdir'], "config.json"), 'w') as f:
        json.dump(out_dict, f, separators=(',', ': '))


def sigterm_handler(signum, frame):
    print("hander called with signal number: {}".format(signum))
    raise SIGTERMERROR("sigterm received")

def test_ldpc(args, dm):
    """
    Test trained GNN on LDPC test dataset
    """
    # system.load_from_checkpoint(args['checkpoint_path'], model_kwargs=model_kwargs, control_kwargs=control_kwargs)
    # device = system.device
    device = args['device']
    system = GNNSystem.load_from_checkpoint(args['checkpoint_path'])
    system.to(device)
    test_loader = dm.test_dataloader()
    
    N = 96
    M = 48
    # the following code is copied and modifed from FGNN codebase: https://github.com/zzhang1987/Factor-Graph-Neural-Network
    acc_seq = []
    acc_cnt = np.zeros((5, 6))
    acc_tot = np.zeros((5, 6))
    tot = 0
    system.eval()
    # pdb.set_trace()
    SNR = [0, 1, 2, 3, 4]
    for g in tqdm(test_loader):
        g = g.to(next(system.model.parameters()).device)
        cur_SNR = g.x2[:,0].contiguous().view(-1,N)[:,0]  # (B,)
        sigma_b = g.x2[:,1].contiguous().view(-1,N)[:,0]
        g.x2 = g.x2[:,2:]

        yhat = system.model.forward(g,output_dynamics=False, nstep=system.get_nstep())
        yhat = yhat.reshape(-1,N)
        label = g.y.reshape(-1,N)
        B = label.shape[0]

        pred_int = (yhat >= 0).long().squeeze()  # (B,N)
        label = label.squeeze()  # (B,N)

        for csnr in SNR:
            for b in range(6):
                indice = (sigma_b.long() == b) & (abs(cur_SNR - csnr) < 1e-3)
                acc_cnt[csnr][b] += torch.sum(pred_int[indice, :48]
                                              == label[indice, :48]).item()
                acc_tot[csnr][b] += torch.sum(indice) * 48

        all_correct = torch.sum(pred_int[:, :48] == label[:, :48])

        acc_seq.append(all_correct.item())
        tot += np.prod(label.shape) // 2

    print(1 - sum(acc_seq) / tot)
    err_class = 1 - np.divide(acc_cnt, acc_tot)
    print(torch.FloatTensor(err_class))

signal.signal(signal.SIGTERM, sigterm_handler)


@ex.automain
def run(args, dataset, model_args):
    path_init()

    model_kwargs = {
        'nin1':dataset['n1'],
        'nin2':dataset['n2'],
        'nout1':None,
        'nout2':dataset['ntarget'],
        'total_factor_types':dataset['total_factor_types']
    }
    model_kwargs.update(**model_args)
    control_keys = [
        'nstep_start',
        'nstep_end',
        'nstep_schedule_method',
        'nstep_linear_epochs',
        'burnin_epochs',
        'init_lr',
        'max_lr',
        'gamma',
        'epochs',
        'optimizer_name',
        'scheduler_name',
        'scheduler_patience',
        'plot_interval',
        'loss_fn',
    ]
    control_kwargs = {k:args[k] for k in control_keys}

    ## Load dataset 
    argnames = list(inspect.signature(load_data).parameters.keys())
    dataset_args = {k:dataset[k] for k in argnames}
    data_dm = FactorGraphDataModule(dataset_args)

    print(f"Model config:\n {model_kwargs}")
    print('-----------------------------')
    print(f"Control config:\n{control_kwargs}")
    system = GNNSystem(
        model_kwargs, control_kwargs, architecture=args["architecture"],
    )
    gpus = torch.cuda.device_count() if args["use_cuda"] and torch.cuda.is_available() else 0
    # debug info incase of NCCL error: https://github.com/PyTorchLightning/pytorch-lightning/issues/4420
    print(f"Using {gpus} GPUS")
    device = "cuda:0" if args["use_cuda"] and torch.cuda.is_available() else "cpu"
    system.to(device)

    # callbacks 
    if args["architecture"] in ['cnf', 'cnf2']:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args["realoutdir"],
            filename='{epoch}-{val_loss_epoch:.5f}',
            monitor='val_loss_epoch',
            mode='min',
            period=1,
            save_top_k=-1,  # save all models
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args["realoutdir"],
            filename='{epoch}-{val_r2:.5f}',
            monitor='val_r2',
            mode='max',
            period=1,
            save_top_k=-1,  # save all models
        )
    early_stop_callback = EarlyStopping(
        monitor='val_loss_epoch',
        min_delta=0.00,
        patience=args['patience'],
        verbose=False,
        mode='min'
    )
    # logger 
    logger = None
    trainer = pl.Trainer(
        min_epochs=1, max_epochs=args["epochs"],
        gpus=gpus,
        log_every_n_steps=3,
        flush_logs_every_n_steps=50,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator='ddp'
    )

    ## Train!
    trainer.fit(system, data_dm)
