# factor-inference-net

codebase for factor inference network

## Enviroments

- `Python 3.8`

- `PyTorch 1.8.0`: please follow instructions on [https://pytorch.org/](https://pytorch.org/) to install PyTorch based on your OS and hardware

- `PyTorch Geometric 1.7.0`: please follow instructions on [https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to install Pytorch Geometric based on your OS and hardware

Other packages can be installed via:

```bash
pip install -r requirements.txt
```

## Generate datasets

### Gaussian 

Commands to generate train and test datasets of Gaussian tree graphs:

```bash
# Training set
python scripts/data/gaussian.py --outdir=data/gaussian/tree/train/n10 --structure=tree --ndata=30000 --nnode=10
# Testing sets
for i in `seq 10 10 50`; do
    python scripts/data/gaussian.py --outdir=data/gaussian/tree/test/n$i --structure=tree --ndata=2000 --nnode=$i
done
```

Commands to generate train and test datasets of general Gaussian graphical models (GGM) with various graph structures:

```bash
# Training set
python scripts/data/gaussian.py --outdir=data/gaussian/all/train/n10 --structure=all --ndata=10000 --nnode=10
# Testing sets
for i in `seq 10 10 50`; do
    python scripts/data/gaussian.py --outdir=data/gaussian/all/test/n$i --structure=all --ndata=2000 --nnode=$i
done
```

### Discrete spin-glass systems with third-order interactions

Commands to generate train and test datasets

```bash
julia --project -p 8 scripts/data/3spin.jl --outdir=data/third/discrete/train/n10 --gamma=0.5 --ndata=10000 --n=10
for i in `seq 5 1 15`; do
    julia --project -p 8 scripts/data/3spin.jl --outdir=data/third/discrete/test/n$i --gamma=0.5 --ndata=1000 --n=$i
done
```

### Continuous third-order graphical models

Commands to generate train and test datasets

```bash
## Training dataset
# firsting store the graph structure, coefficients, along with the MCMC samples
julia --project -p 8 scripts/data/3cont-generate_graphs-1.jl --outdir=data/third/continuous/train/n10/samples --n=10 --ndata=10000
# then calculate whatever statistics (first 4 moments in our experiments) based on the samples and store the (graph, moments) as a dataset for GNN
julia --project -p 8 scripts/data/3cont-make_dataset-2.jl --indir=data/third/continuous/train/n10/samples --outdir=data/third/continuous/train/n10/raw --order=4 --statistics=central_moments

## Testing datasets
for i in `seq 10 10 20`; do
    julia --project -p 8 scripts/data/3cont-generate_graphs-1.jl --outdir=data/third/continuous/test/n$i/samples --n=$i --ndata=10000
    julia --project -p 8 scripts/data/3cont-make_dataset-2.jl --indir=data/third/continuous/test/n$i/samples --outdir=data/third/continuous/test/n$i/raw --order=4 --statistics=central_moments
done
```

## Train GNNs

### Gaussian datasets

* Train a recurrent Factor-GNN on the Tree GGM dataset. For a stacked version just replace `--architecture=rnn` with `--architecture=rnn` and add `--model_args.nlayer=10`. This works for all datasets

```bash
python train_lightning.py with dataset.datatype=gaussianbipartite dataset.dataroot=data/gaussian/tree/train dataset.dataset=n10 dataset.batch_size=100 args.epochs=1000 args.outdir=outputs/gaussian/tree args.architecture=rnn args.loss_fn=mse args.nstep_start=30 args.nstep_end=50 args.nstep_schedule_method=random args.init_lr=1e-3 args.patience=50 args.scheduler_patience=25 model_args.heads=5
```

* Train a GNN on the loopy GMM dataset

```bash
python train_lightning.py with dataset.datatype=gaussianbipartite dataset.dataroot=data/gaussian/all/train dataset.dataset=n10 dataset.batch_size=100 args.epochs=1000 args.outdir=outputs/gaussian/tree args.architecture=rnn args.loss_fn=bce args.nstep_start=30 args.nstep_end=50 args.nstep_schedule_method=random args.init_lr=1e-3 args.patience=50 args.scheduler_patience=25 model_args.heads=5
```

### Discrete spin-glass dataset

```bash
python train_lightning.py with dataset.datatype=gaussianbipartite dataset.dataroot=data/third/discrete/train dataset.dataset=n10 dataset.batch_size=100 args.epochs=1000 args.outdir=outputs/gaussian/tree args.architecture=rnn args.loss_fn=mse args.nstep_start=30 args.nstep_end=50 args.nstep_schedule_method=random args.init_lr=1e-3 args.patience=50 args.scheduler_patience=25 model_args.heads=5
```

### Continuous third-order dataset

```bash
python train_lightning.py with dataset.datatype=gaussianbipartite dataset.dataroot=data/third/continuous/train dataset.dataset=n10 dataset.batch_size=100 args.epochs=1000 args.outdir=outputs/gaussian/tree args.architecture=rnn args.loss_fn=mse args.nstep_start=30 args.nstep_end=50 args.nstep_schedule_method=random args.init_lr=1e-3 args.patience=50 args.scheduler_patience=25 model_args.heads=5
```